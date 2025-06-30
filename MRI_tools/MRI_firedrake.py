from typing import Optional, List, Tuple
import numpy as np
from mpi4py import MPI
from scipy.ndimage import binary_dilation
import firedrake as fd
from firedrake.__future__ import interpolate
from MRI_tools.MRI_base import MRIBase, add_timing


class MRI(MRIBase):
    """A class taking care of MRI related stuff for firedrake

    Args:
        size: size of MRI grid
        origin: location of center of voxel (0, 0, 0)
        spacing: width, height and depth of a single voxel
        rotation: rotation matrix 3x3 to get from voxel coordinates to world coordinates
        timestep: timestep duration in seconds
        number_of_timesteps: number of timesteps of MR image
        venc: velocity encoding parameter - maximal attainable velocity
    """

    @add_timing
    def compute_roi(self, mesh: fd.MeshGeometry, padding=0) -> Tuple[np.ndarray]:
        """find corners of a box mesh (subset of given MRI image) that contains given computational mesh

        Region of interest (ROI) is a rectangular part of MRI mesh that contains provided mesh

        Args:
            mesh: computational mesh around which should the boxmesh be created

        Returns: lcorner, rcorner - numpy arrays of box corners in ijk coordinates
        """
        lbound = np.zeros(3)
        rbound = self.size - np.ones(3)
        mesh_coords = mesh.coordinates.dat.data_ro
        ijk_coords = np.array([self.xyz_to_ijk(c) for c in mesh_coords])
        lcorner = np.array([round(ijk_coords[:, i].min()) for i in range(3)]).astype(
            float
        ) - padding * np.ones(3)
        rcorner = np.array([round(ijk_coords[:, i].max()) for i in range(3)]).astype(
            float
        ) + padding * np.ones(3)
        np.clip(lcorner, lbound, rbound, out=lcorner)
        np.clip(rcorner, lbound, rbound, out=rcorner)
        glob_lcorner = [mesh.mpi_comm().allreduce(coord, MPI.MIN) for coord in lcorner]
        glob_rcorner = [mesh.mpi_comm().allreduce(coord, MPI.MAX) for coord in rcorner]
        if mesh.mpi_comm().rank == 0:
            print(f"ROI found between {glob_lcorner} and {glob_rcorner}.")
        return np.array(glob_lcorner, dtype=np.int16), np.array(
            glob_rcorner, dtype=np.int16
        )

    @add_timing
    def create_mri_space(
        self,
        lcorner: Optional[np.ndarray] = None,
        rcorner: Optional[np.ndarray] = None,
        hexahedral: bool = True,
        space_type: str = "DG",
        comm=fd.COMM_WORLD,
    ):
        """Create box mesh using MRI origin, spacing and rotation in world coordinates
        Corners of the box can be specified as lcorner and rcorner using ijk coordinates

        Args:
            lcorner: default: (0, 0, 0), lower bound of MRI voxels
            rcorner: default: (size_x-1, size_y-1, size_z-1), upper bound for MRI voxels
            hexahedral: default: True, whether the mesh should be hexahedral (or tetrahedral)
            space_type: default: DG, define if the values should be kept in voxel volumes (DG/Discontinuous Lagrange) or vertices (CG, Lagrange)
            comm: default: COMM_WORLD, MPI communicator, which divides the mesh between processors (has major effect in parallel)

        Return:
            boxmesh, mri_space - created boxmesh and the appropriate function space defined on it
        """
        if space_type in ["DG", "Discontinuous Lagrange"]:
            degree = 0
        elif space_type in ["CG", "Lagrange"]:
            degree = 1
        else:
            raise ValueError(f"{space_type} MRI function space not supported.")
        # deal with defaults
        if lcorner is None:
            lcorner = np.zeros(3, dtype=np.int8)
        if rcorner is None:
            if degree == 0:
                rcorner = np.array(self.size)
            else:
                rcorner = np.array(self.size) - np.ones(3, dtype=np.int8)

        # figure out the degree of the function space
        # compute the size of box mesh needed and create it
        boxsize = tuple(rcorner - lcorner)
        boxmesh = fd.BoxMesh(*boxsize, *boxsize, hexahedral=hexahedral, comm=comm)
        # if DG space, the mesh has to be shifted in order to have centers of voxels at gridpoints
        if degree == 0:
            shift = lcorner - 0.5 * np.ones(3)
        elif degree == 1:
            shift = lcorner
        for i, sh in enumerate(shift):
            boxmesh.coordinates.dat.data[:, i] += sh
        # transform mesh to xyz coordinates
        V = boxmesh.coordinates.function_space()
        x, y, z = fd.SpatialCoordinate(boxmesh)
        ijk = self.ijk_to_xyz(np.array([x, y, z]))
        transform = fd.Function(V).interpolate(fd.as_vector(ijk))
        boxmesh.coordinates.assign(transform)
        boxmesh.clear_spatial_index()
        if self.dim == 3:
            function_space = fd.VectorFunctionSpace(boxmesh, space_type, degree)
        elif self.dim == 1:
            function_space = fd.FunctionSpace(boxmesh, space_type, degree)
        else:
            raise ValueError(f"Data has invalid spatial dimension: {self.dim}")
        return function_space

    @add_timing
    def data_to_mesh(
        self,
        function_space: Optional = None,
        data: Optional[np.ndarray] = None,
        timesteps: Optional[List[int]] = None,
    ) -> List[fd.Function]:
        """Read data from numpy array to mri_space defined on boxmesh to create a list of functions from data at provided timesteps

        Args:
            function_space: a function space into which the data should be loaded (default: DG0 on hexahedral mesh of the size of the MRI image)
            data: numpy array of data to be loaded to mri_space function (default: data read during initialization of MRI object)
            timesteps: list of integer timesteps which should be loaded (default: list of all timesteps)

        Returns: List of functions which contain the data loaded to mri_space at given timesteps
        """
        if data is None:
            data = self.data
        if function_space is None:
            function_space = self.create_mri_space()
        if timesteps is None:
            timesteps = list(range(self.number_of_timesteps))
        ijk_points = self._dof_coordinates_to_ijk(function_space)
        functions = []
        for i in timesteps:
            f = fd.Function(function_space, name="mri")
            if self.dim == 3:
                data_values = [
                    np.dot(self.rotation, data[i, :, ijk[0], ijk[1], ijk[2]])
                    for ijk in ijk_points
                ]
            elif self.dim == 1:
                data_values = [data[i, 0, ijk[0], ijk[1], ijk[2]] for ijk in ijk_points]
            f.dat.data[:] = data_values
            functions.append(f)
        return functions

    @add_timing
    def extract_array_from_mri_functions(
        self, functions: List[fd.Function]
    ) -> np.ndarray:
        """Takes a list of functions and returns the numpy array of shape (number_or_timesteps, 3, *size)
        The values outside the parent mesh provided to the vertex only mesh will be set to zero

        Args:
            functions: list of dg0 functions defined on a vertex only mesh (distributed the same way as the parent mesh)

        Returns: numpy array - summed over all the processors
        """
        function_space = functions[0].function_space()
        dim = function_space.value_size
        # comm = function_space.mesh().mpi_comm()
        np_array = np.zeros((len(functions), dim, *self.size))
        ijk_points = self._dof_coordinates_to_ijk(function_space)
        for i, f in enumerate(functions):
            datapoints = f.dat.data_ro
            for dat, ijk in zip(datapoints, ijk_points):
                if dim == 3:
                    np_array[i, :, ijk[0], ijk[1], ijk[2]] = np.dot(
                        self.inv_rotation, dat
                    ).round(
                        decimals=2
                    )  # round to cm/s

                elif dim == 1:
                    np_array[i, 0, ijk[0], ijk[1], ijk[2]] = dat

        # functions are distributed -> sum all the arrays
        # glob_array = comm.allreduce(np_array, MPI.SUM)
        return np_array

    @add_timing
    def make_mask(self, fem_mesh: fd.MeshGeometry, mri_space = None, padding=0):
        """get DG0 function on mri mesh containing information on how much the fem_mesh intersects each voxel

        Args:
            fem_mesh: computational mesh
            mri_space: CG1 or DG0 space on a hexahedral mesh representing the MRI grid
        """
        if mri_space is None:
            mri_space = self.create_mri_space()
        el = fd.FiniteElement("CG", fem_mesh.ufl_cell(), 1)
        Q = fd.FunctionSpace(fem_mesh, el)
        cont_one = fd.assemble(interpolate(fd.Constant(1.0), Q))
        #projected = fd.project(cont_one, mri_space)
        projected = fd.assemble(interpolate(cont_one, mri_space, allow_missing_dofs=True))
        mask = self.extract_array_from_mri_functions([projected])
        if padding > 0:
            # Convert to binary mask
            binary_mask = mask > 0.0
            binary_mask = binary_dilation(binary_mask, iterations=padding)
            mask = binary_mask.astype(np.uint8)
        return mask

    def _dof_coordinates_to_ijk(self, function_space):
        """Given a function space, return a list of its degrees of freedom in ijk coordinates

        Args:
            function_space: a function space whose dofs are to be transformed

        Returns: List of dofs in ijk coordinates in the original order
        """
        dim = function_space.value_size
        mesh = function_space.mesh()
        if dim == 3:
            # get coordinates of dofs
            xyz_function = fd.assemble(interpolate(mesh.coordinates, function_space))
            xyz_points = xyz_function.dat.data_ro
        elif dim == 1:
            # interpolate coordinates one by one to the scalar function space
            xyz_functions = [
                fd.assemble(interpolate(mesh.coordinates[i], function_space))
                for i in range(3)
            ]
            # get list of x, y and z coordinations of dofs and rearrange them to list of dof coordinates
            xyz_points = np.array(
                [xyz_functions[i].dat.data_ro for i in range(3)]
            ).transpose()
        else:
            raise ValueError(f"Invalid value size of the function space: {dim}")
        # transform xyz coordinates to ijk
        ijk_points = [self.xyz_to_ijk(xyz) for xyz in xyz_points]
        return ijk_points
