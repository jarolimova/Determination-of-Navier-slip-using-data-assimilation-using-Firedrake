"""This module implements various finite elements which can be used to solve the Navier Stokes equations using FEM
string_to_element dict gives a mapping between strings and element classes to avoid calling getattr: element = string_to_element["th"](mesh, bndry)
"""

from abc import ABC
import firedrake as fd

__all__ = [
    "P1P1",
    "TaylorHood",
    "P2P0",
    "MiniElement",
    "ConformingCrouzeixRaviart",
    "NonconformingCrouzeixRaviart",
    "ScottVogelius",
    "string_to_element",
]


class NSElement(ABC):
    """Abstract base class for the finite element objects"""

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        """Initialize the finite element object

        Args:
            mesh: FEM computational mesh (fd.Mesh object)
            bndry: mesh function with boundary marks
        """
        self.mesh = mesh
        self.c = mesh.ufl_cell()
        self.stable = True
        self.W = None

    def split(self, w: fd.Function, test: bool = True):
        t = fd.split(w)
        v = t[0]
        p = t[1]
        if test:
            t_ = fd.TestFunctions(self.W)
            v_ = t_[0]
            p_ = t_[1]
            return (v, p, v_, p_)
        else:
            return (v, p)

    def extract(self, w: fd.Function):
        v, p = w.subfunctions
        return (v, p)


class P1P1(NSElement):
    """P1P1 element - piecewise linear velocities and pressures
    not stable element (does not satisfy the inf-sup condition)
    need stabilization to work properly: usually supg, ip in v and p works as well
    number of dofs: v_dofs = dim * #points, p_dofs = #points
    """

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        super().__init__(mesh)
        V = fd.VectorElement("CG", self.c, 1)
        P = fd.FiniteElement("CG", self.c, 1)
        E = fd.MixedElement([V, P])
        self.W = fd.FunctionSpace(mesh, E)
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)
        self.stable = False


class TaylorHood(NSElement):
    """Taylor Hood element - piecewise quadratic in velocities and piecewise linear in pressures
    stable element (satisfy inf-sup condition)
    standart element for incompressible Navier Stokes
    number of dofs: v_dofs = dim * (#points + #faces), p_dofs = #points
    """

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        super().__init__(mesh)
        V = fd.VectorElement("CG", self.c, 2)
        P = fd.FiniteElement("CG", self.c, 1)
        E = fd.MixedElement([V, P])
        self.W = fd.FunctionSpace(mesh, E)
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)


class P2P0(NSElement):
    """P2P0 element - piecewise quadratic in velocities and piecewise constant in pressures
    stable element (satisfy inf-sup)
    advantage: better incompressibility, disadvantage: more expensive and less acturate than T-H in pressure
    number of dofs: v_dofs = dim * (#points + #faces), p_dofs = #cells
    """

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        super().__init__(mesh)

        V = fd.VectorElement("CG", self.c, 2)
        P = fd.FiniteElement("DG", self.c, 0)
        E = fd.MixedElement([V, P])
        self.W = fd.FunctionSpace(mesh, E)
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)


class MiniElement(NSElement):
    """Mini Element - piecewise linear + bubble in velocities and piecewise linear in pressures
    stable element (satisfy inf-sup condition)
    standart element for incompressible Navier Stokes
    number of dofs: v_dofs = dim * (#points + #cells), p_dofs = #points
    """

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        super().__init__(mesh)

        d = mesh.topological_dimension()
        U = fd.FiniteElement("CG", self.c, 1)
        B = fd.FiniteElement("Bubble", self.c, d + 1)
        P = fd.FiniteElement("CG", self.c, 1)
        V = fd.VectorElement(fd.NodalEnrichedElement(U, B))
        E = fd.MixedElement([V, P])
        self.W = fd.FunctionSpace(mesh, E)
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)


class ConformingCrouzeixRaviart(NSElement):
    """Conforming Crouzeix Raviart element - piecewise quadratic + bubble in velocities and linear discontinuous in pressures
    stable element (satisfy inf-sup condition)
    number of dofs: v_dofs = dim * (#points + #faces + #cells), p_dofs = (dim + 1) * #cells
    """

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        super().__init__(mesh)

        d = mesh.topological_dimension()
        U = fd.FiniteElement("CG", self.c, 2)
        B = fd.FiniteElement("Bubble", self.c, d + 1)
        P = fd.FiniteElement("DG", self.c, 1)
        V = fd.VectorElement(fd.NodalEnrichedElement(U, B))
        E = fd.MixedElement([V, P])
        self.W = fd.FunctionSpace(mesh, E)
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)


class NonconformingCrouzeixRaviart(NSElement):
    """Nonconforming Crouzeix Raviart element - discontinuous linear in velocities and piecewise constant in pressures
    stable element (satisfy inf-sup condition)
    advantage: relatively cheap, discontinuous, disadvantage: low stability with convection
    number of dofs: v_dofs = dim * #faces, p_dofs = #cells
    """

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        super().__init__(mesh)

        V = fd.VectorElement("CR", self.c, 1)
        P = fd.FiniteElement("DG", self.c, 0)
        E = fd.MixedElement([V, P])
        self.W = fd.FunctionSpace(mesh, E)
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)


class ScottVogelius(NSElement):
    """Scott Vogelius element - 4th order in velocities and 3th order in pressures
    stable element (satisfy inf-sup condition)
    advantage: pointwise divergergence free, disadvantage: most expensive
    """

    def __init__(self, mesh: fd.Mesh, *args, **kwargs):
        super().__init__(mesh)

        V = fd.VectorElement("CG", self.c, 4)
        P = fd.FiniteElement("DG", self.c, 3)
        E = fd.MixedElement([V, P])
        self.W = fd.FunctionSpace(mesh, E)
        self.V = self.W.sub(0)
        self.P = self.W.sub(1)


string_to_element = {
    "p1p1": P1P1,
    "th": TaylorHood,
    "p2p0": P2P0,
    "mini": MiniElement,
    "ccr": ConformingCrouzeixRaviart,
    "ncr": NonconformingCrouzeixRaviart,
    "sv": ScottVogelius,
}
