import firedrake as fd
import firedrake.adjoint as fda
from firedrake.__future__ import interpolate
from forward import AnalyticInletProfile, marks, l2_norm_with_optional_interpolation
from petsc4py import PETSc

print = PETSc.Sys.Print


class ErrorFunctional:
    """Error functional for the steady optimization problem in the assimilation process.

    Args:
        data: The data to be used for the error functional.
        v_mag: Magnitude of the velocity.
        mu: Dynamic viscosity.
        in_data: Input data for the inlet profile.
        mesh: The mesh on which the problem is defined.
        V: Function space for the velocity field.
        slip_penalty: Penalty for slip boundary conditions.
        alpha (float, optional): Weight for the inlet profile error term. Defaults to 0.0.
        beta (float, optional): Weight for the wall control error term. Defaults to 0.0.
        gamma (float, optional): Weight for the wall control error term. Defaults to 0.0.
        char_len (float, optional): Characteristic length scale for the problem. Defaults to 1.0.
        ds (function, optional): Measure for the inlet boundary. Defaults to fd.ds.
        dx (function, optional): Measure for the domain. Defaults to fd.dx.
    """

    def __init__(
        self,
        data,
        v_mag,
        mu,
        in_data,
        mesh,
        V,
        slip_penalty,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        char_len=1.0,
        ds=fd.ds,
        dx=fd.dx,
    ):
        self.data = data
        self.v_mag = v_mag
        self.in_data = in_data
        self.mu = mu
        self.mesh = mesh
        self.V = V
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.char_len = char_len
        self.profile = AnalyticInletProfile(
            self.in_data, self.mesh, self.mu, slip_penalty
        )
        self.ds = ds
        self.dx = dx

    def __call__(self, u, wall_control, u_in):
        """Compute the error functional value for the given velocity field, wall control, and inlet profile."""
        J = self.l2_dx(u)
        print("l2_dx: ", J)
        J += self.alpha * self.grad_l2_uin_ds(wall_control, u_in)
        J += self.beta * self.grad_l2_wall_control_ds(wall_control)
        J += self.gamma * self.l2_ds(wall_control, u_in)
        return J

    def l2_dx(self, u):
        """Compute the L2 norm of the velocity field over the domain."""
        l = self.char_len
        dat = self.data
        return 0.5 / pow(l, 3) * l2_norm_with_optional_interpolation(u, dat)

    def l2_ds(self, wall_control, u_in):
        """Compute the L2 norm of the difference between the inlet profile and the inlet velocity field over the inlet boundary."""
        l = self.char_len
        profile = self.profile(wall_control, self.v_mag)
        return fd.assemble(
            0.5
            / pow(l, 2)
            * fd.inner(
                u_in - profile,
                u_in - profile,
            )
            * self.ds(marks["in"])
        )

    def grad_l2_uin_ds(self, wall_control, u_in):
        """Compute the gradient of the L2 norm of the inlet velocity field over the inlet boundary."""
        mesh = u_in.function_space().mesh()
        identity = fd.Identity(3)
        n = fd.FacetNormal(mesh)
        It = identity - fd.outer(n, n)
        return fd.assemble(
            0.5
            * fd.inner(fd.grad(u_in) * It, fd.grad(u_in) * It)
            * self.ds(marks["in"])
        )

    def grad_l2_wall_control_ds(self, wall_control):
        """Compute the gradient of the L2 norm of the wall control over the wall boundary."""
        try:
            mesh = wall_control.function_space().mesh()
            identity = fd.Identity(3)
            n = fd.FacetNormal(mesh)
            It = identity - fd.outer(n, n)
            return fd.assemble(
                0.5
                * fd.inner(It * fd.grad(wall_control), It * fd.grad(wall_control))
                * self.ds(marks["wall"])
            )
        except ValueError:
            return 0.0


class ErrorFunctionalUnsteady:
    """Error functional for the unsteady optimization problem in the assimilation process.
    Args:
        mesh: The mesh on which the problem is defined.
        dt: Time step size.
        T: Total simulation time. (characteristic time scale for the problem)
        char_len: Characteristic length scale for the problem.
    """

    def __init__(self, mesh, dt, T, char_len):
        self.mesh = mesh
        self.dt = dt
        self.T = T
        self.char_len = char_len

    def grad_reg(self, u_in_list):
        """Compute the gradient regularization term for the unsteady optimization problem."""
        grad_reg = 0.0
        n = fd.FacetNormal(self.mesh)
        It = fd.Identity(3) - fd.outer(n, n)
        grad_reg_list = [
            fd.assemble(
                fd.inner(fd.grad(u_in) * It, fd.grad(u_in) * It) * fd.ds(marks["in"])
            )
            for u_in in u_in_list
        ]
        grad_reg = trapezoidal_rule(grad_reg_list, self.dt)
        return grad_reg / (2.0 * self.T)

    def time_reg(self, u_in_list):
        """Compute the time regularization term for the unsteady optimization problem."""
        time_reg = 0.0
        time_reg_list = [
            fd.assemble(
                fd.inner(
                    u_in_list[i] - u_in_list[i - 1], u_in_list[i] - u_in_list[i - 1]
                )
                / pow(self.dt, 2)
                * fd.ds(marks["in"])
            )
            for i in range(1, len(u_in_list))
        ]
        time_reg = trapezoidal_rule(time_reg_list, self.dt)
        return time_reg * self.T / (2.0 * pow(self.char_len, 2))

    def timegrad_reg(self, u_in_list):
        """Compute the time gradient regularization term for the unsteady optimization problem."""
        timegrad_reg = 0.0
        timegrad_reg_list = [
            fd.assemble(
                fd.inner(
                    fd.grad(u_in_list[i]) - fd.grad(u_in_list[i - 1]),
                    fd.grad(u_in_list[i]) - fd.grad(u_in_list[i - 1]),
                )
                / pow(self.dt, 2)
                * fd.ds(marks["in"])
            )
            for i in range(1, len(u_in_list))
        ]
        timegrad_reg = trapezoidal_rule(timegrad_reg_list, self.dt)
        return timegrad_reg * self.T / 2.0

    def grad_l2_wall_control_ds(self, wall_control):
        """Compute the gradient of the L2 norm of the wall control over the wall boundary."""
        try:
            mesh = wall_control.function_space().mesh()
            identity = fd.Identity(3)
            n = fd.FacetNormal(mesh)
            It = identity - fd.outer(n, n)
            return fd.assemble(
                0.5
                * fd.inner(It * fd.grad(wall_control), It * fd.grad(wall_control))
                * fd.ds(marks["wall"])
            )
        except ValueError:
            return 0.0

    def kinetic_energy(self, u_in_list):
        kinetic_list = [
            fd.assemble(fd.inner(uin, uin) * fd.ds(marks["in"])) for uin in u_in_list
        ]
        kinetic_int = trapezoidal_rule(kinetic_list, self.dt)
        return kinetic_int / (2.0 * self.T * pow(self.char_len, 2))


def trapezoidal_rule(f_list, dt):
    f_int = 0.0
    for i in range(1, len(f_list)):
        f_int += 0.5 * (f_list[i - 1] + f_list[i]) * dt
    return f_int
