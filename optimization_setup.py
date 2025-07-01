import firedrake as fd
from forward import marks
from petsc4py import PETSc

print = PETSc.Sys.Print


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
