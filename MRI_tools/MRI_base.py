import os
import time
import json
import getpass
from textwrap import dedent
from functools import wraps
from typing import List, Optional
from fnmatch import fnmatch
import vtk
import vtk.util.numpy_support as numpy_support
import numpy as np
import pydicom
import SimpleITK as sitk
from mpi4py import MPI


def add_timing(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Initialized {func.__name__} ...")
        start = time.time()
        output = func(*args, **kwargs)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"{func.__name__} finished. (time[s]: {time.time()-start})")
        return output

    return decorated


class MRIBase:
    """Base class containing mostly initializations from MRI data and some useful functionality
    helps to separate FEniCS or firedrake code from the shared parts

    Args:
        size: size of MRI grid
        origin: location of center of voxel (0, 0, 0)
        spacing: width, height and depth of a single voxel
        rotation: rotation matrix 3x3 to get from voxel coordinates to world coordinates
        timestep: timestep duration in seconds
        number_of_timesteps: number of timesteps of MR image
        venc: velocity encoding parameter - maximal attainable velocity
        scale: scale shades of gray to velocity in m/s
    """

    def __init__(
        self,
        size: np.ndarray = np.ones(3, dtype=np.int8),
        origin: np.ndarray = np.zeros(3),
        spacing: np.ndarray = np.ones(3),
        rotation: np.ndarray = np.identity(3),
        timestep: float = 0.1,
        number_of_timesteps: int = 1,
        data: Optional[np.ndarray] = None,
        venc: Optional[float] = None,
        scale: float = 1.0,
    ):
        self.results_folder = f"/usr/work/{getpass.getuser()}/MRI_simulation_results"
        self.origin = np.array(origin)
        self.spacing = np.array(spacing)
        self.rotation = np.array(rotation)
        self.timestep = float(timestep)
        self.venc = venc
        if data is None:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("Missing input data! Puting zeros instead.")
            self.data = np.zeros((number_of_timesteps, 3, *size))
        else:
            self.data = scale * np.array(data)
            if (
                (len(self.data.shape) != 5)
                or (self.data.shape[0] != number_of_timesteps)
                or (self.data.shape[2:] != np.array(size)).any()
            ):
                raise ValueError(
                    f"Wrong data shape: {self.data.shape}\n"
                    + "Desired shape is: "
                    + "(number_of_timesteps, number_of_directions, i_voxels, j_voxels, k_voxels)"
                )
        if MPI.COMM_WORLD.Get_rank() == 0:
            rotation_str = "".join(
                np.array2string(
                    self.rotation, separator=", ", suppress_small=True
                ).split("\n")
            )
            print(
                "Initialized MRI object with:\n",
                f"size: {np.array(size)} => {np.array(size).prod()} voxels,\n",
                f"number_of_timesteps: {number_of_timesteps},\n",
                f"spacing: {self.spacing},\n",
                f"origin: {self.origin},\n",
                f"timestep: {self.timestep},\n",
                f"VENC: {self.venc},\n",
                f"rotation: {rotation_str},\n",
            )

    def __str__(self):
        rotation_str = "".join(
            np.array2string(self.rotation, separator=", ", suppress_small=True).split(
                "\n"
            )
        )
        return dedent(
            f"""
            size: {np.array(self.size)} => {np.array(self.size).prod()} voxels,
            number_of_timesteps: {self.number_of_timesteps},
            spacing: {self.spacing},
            origin: {self.origin},
            timestep: {self.timestep},
            VENC: {self.venc},
            rotation: {rotation_str}\n
            """
        ).strip()

    def __eq__(self, other):
        if (np.array(self.rotation) != np.array(other.rotation)).any():
            print("different rotation: ", self.rotation, other.rotation)
            return False
        elif (np.array(self.origin) != np.array(other.origin)).any():
            print("different origin: ", self.origin, other.origin)
            return False
        elif (np.array(self.size) != np.array(other.size)).any():
            print("different size: ", self.size, other.size)
            return False
        elif (np.array(self.spacing) != np.array(other.spacing)).any():
            print("different spacing: ", self.spacing, other.spacing)
            return False
        elif self.timestep != other.timestep:
            print("different timestep: ", self.timestep, other.timestep)
            return False
        elif self.number_of_timesteps != other.number_of_timesteps:
            print(
                "different number_of_timesteps: ",
                self.number_of_timesteps,
                other.number_of_timesteps,
            )
            return False
        elif (np.array(self.data) != np.array(other.data)).any():
            print("different data")
            return False
        else:
            return True

    @classmethod
    def from_mhd(cls, filenames: List[str], **other_parameters):
        """Initialize instance of MRIBase from MRI image in mhd format to mimic the same setting

        Sets size, origin, spacing, rotation, timestep, number_of_timesteps
        Args:
            filename: path to a file containing MRI image readable by SimpleITK
            other parameters used for initialization
        """
        dim = 4
        itk_images = [sitk.ReadImage(name) for name in filenames]
        itk_arrays = [sitk.GetArrayFromImage(image) for image in itk_images]
        # fix the case when the data is not 4D but 3D
        if itk_arrays[0].ndim == 3:
            itk_arrays = [[array] for array in itk_arrays]
            dim = 3
        elif itk_arrays[0].ndim != 4:
            raise NotImplementedError(
                "provided files does not contain 3D or 4D data, other dimensions are not implemented"
            )
        n_timesteps = len(itk_arrays[0])

        # Transpose the data so we can access the voxel values the correct way (i.e. data[i,j,k])
        # reoder and change orientation of velocity components to match th coordinate system
        data = np.array(
            [
                [array[t].transpose(2, 1, 0) for array in itk_arrays]
                for t in range(n_timesteps)
            ]
        )
        if data.shape[1] == 3:
            data = np.array(
                [[directions[1], -directions[0], -directions[2]] for directions in data]
            )

        RAS_to_LPI = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        # Read data from files
        itk_spacing = np.array(itk_images[0].GetSpacing())
        itk_origin = np.array(itk_images[0].GetOrigin())
        itk_rotation = np.array(itk_images[0].GetDirection()).reshape(dim, dim)
        itk_size = np.array(itk_images[0].GetSize())

        # Remove the last entry, which corresponds to the shift in time
        # Multiply spacing and origin by 0.001 to convert from [mm] to [m] units
        # Transform origin and rotation mtx to correct coordinates (from RAS to LPI)
        spacing = 0.001 * itk_spacing[:3]
        origin = np.dot(0.001 * RAS_to_LPI, itk_origin[:3])
        rotation = np.dot(RAS_to_LPI, itk_rotation[:3, :3])
        size = itk_size[:3]
        if dim == 4:
            timestep = itk_spacing[3]
            n_timesteps = itk_size[3]
        else:
            timestep = 0.0
            n_timesteps = 1

        return cls(
            size=size,
            origin=origin,
            spacing=spacing,
            rotation=rotation,
            timestep=timestep,
            number_of_timesteps=n_timesteps,
            data=data,
            **other_parameters,
        )

    @classmethod
    def from_dicom(
        cls, filenames: List[str], datatype="PHASE CONTRAST M", **other_parameters
    ):
        # make lists of the files
        foldernames = [os.path.dirname(filename) for filename in filenames]
        basenames = [os.path.basename(filename) for filename in filenames]
        file_lists = [
            [f for f in os.listdir(foldername) if fnmatch(f, basename)]
            for basename, foldername in zip(basenames, foldernames)
        ]
        list_lengths = [len(file_list) for file_list in file_lists]
        for file_list in file_lists:
            file_list.sort()
        # go through the files and save the slice locations and trigger times
        # dicom_dict connects each file with the specific slice_location and trigger_time
        slice_locations = []
        trigger_times = []
        dicom_dict = dict()
        for file_list, foldername in zip(file_lists, foldernames):
            for filename in file_list:
                dicom_file = pydicom.read_file(f"{foldername}/{filename}")
                if dicom_file[0x0008, 0x0008].value[2] == datatype:
                    slice_location = dicom_file.SliceLocation
                    trigger_time = dicom_file.TriggerTime
                    if slice_location not in slice_locations:
                        slice_locations.append(slice_location)
                    if trigger_time not in trigger_times:
                        trigger_times.append(trigger_time)
                    if slice_location in dicom_dict.keys():
                        if trigger_time in dicom_dict[slice_location].keys():
                            dicom_dict[slice_location][trigger_time].append(filename)
                        else:
                            dicom_dict[slice_location][trigger_time] = [filename]
                    else:
                        dicom_dict[slice_location] = {trigger_time: [filename]}
        # go through the files again and read the data
        data_complete = []
        for trigger_time in trigger_times:
            directions = []
            for i, foldername in enumerate(foldernames):
                ij_data_list = []
                for slice in slice_locations:
                    i_file = dicom_dict[slice][trigger_time][i]
                    dicom_file = pydicom.read_file(f"{foldername}/{i_file}")
                    pc_velocity = dicom_file[0x2001, 0x101A].value
                    venc = max(pc_velocity)
                    rows, columns = int(dicom_file.Rows), int(dicom_file.Columns)
                    image_data = np.zeros((rows, columns), dtype=np.float16)
                    image_data[:, :] = dicom_file.pixel_array
                    image_data -= 2048
                    image_data *= venc / 2048
                    ij_data_list.append(image_data.transpose())
                np_ijk = np.stack(ij_data_list, axis=2)
                directions.append(np_ijk)
            if len(directions) == 3:
                directions[1] *= -1
                directions[2] *= -1
            data_complete.append(directions)
        data = np.array(data_complete)

        # compute the time step size from the trigger_times list
        steps = [
            trigger_times[i + 1] - trigger_times[i]
            for i in range(len(trigger_times) - 1)
        ]
        if len(steps) > 0:
            timestep = 0.001 * sum(steps) / len(steps)
        else:
            timestep = 0.0  # in case there is only one time instant
        nslices = len(slice_locations)
        n_timesteps = len(trigger_times)

        # take the first file and read the general info that is same for all the files
        dicom = pydicom.read_file(f"{foldernames[0]}/{file_lists[0][0]}")
        rows, columns = int(dicom.Rows), int(dicom.Columns)
        dicom_spacing = [
            dicom.PixelSpacing[0],
            dicom.PixelSpacing[1],
            dicom.SpacingBetweenSlices,
        ]
        dicom_origin = dicom.ImagePositionPatient
        dicom_rotation = np.array(dicom.ImageOrientationPatient, dtype=float)
        pc_velocity = dicom[0x2001, 0x101A].value
        venc = max(pc_velocity) / 100
        rotation1 = np.reshape(dicom_rotation, (2, 3))
        rotation2 = [np.cross(rotation1[0], rotation1[1])]
        np_rotation = np.concatenate(
            (rotation1, rotation2)
        ).transpose()  # it is saved using fortran notation for matrices!
        RAS_to_LPI = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        spacing = 0.001 * np.array(dicom_spacing, dtype=float)
        origin = np.dot(RAS_to_LPI, 0.001 * np.array(dicom_origin, dtype=float))
        rotation = np.dot(RAS_to_LPI, np_rotation)
        size = np.array([rows, columns, nslices])
        other_parameters["venc"] = venc
        return cls(
            size=size,
            origin=origin,
            spacing=spacing,
            rotation=rotation,
            timestep=timestep,
            number_of_timesteps=n_timesteps,
            data=data,
            **other_parameters,
        )

    @classmethod
    def from_vti(cls, filenames: List[str], **other_parameters):
        """Initialize instance of MRIBase from MRI image in vti format to mimic the same setting

        Sets size, origin, spacing, rotation, timestep, number_of_timesteps
        Args:
            filename: path to a file containing MRI image readable by SimpleITK
            other parameters used for initialization
        """
        reader = vtk.vtkXMLImageDataReader()
        filename0 = filenames[0].replace("**", "00")
        folder_path = os.path.dirname(filename0)
        with open(folder_path + "/param.txt", "r") as f:
            text = f.read()
            vti_param = [p for p in text.split("\n") if p != ""]
            vti_timesteps = int(vti_param[3])
        # read data
        data = []
        for i in range(vti_timesteps):
            time_string = str(i).zfill(2)
            print("time string", time_string)
            directions = []
            for filename in filenames:
                print(filename)
                reader.SetFileName(filename.replace("**", time_string))
                reader.Update()
                img = reader.GetOutput()
                vti_data = np.reshape(
                    np.array(img.GetPointData().GetArray("scalars")),
                    tuple(reversed(img.GetDimensions())),
                ).transpose(2, 1, 0)
                directions.append(vti_data)
            if len(directions) == 3:
                directions = [-directions[0], directions[1], directions[2]]
            data.append(directions)
        data = np.array(data)

        # read other params
        reader.SetFileName(filename0)
        reader.Update()
        img = reader.GetOutput()
        RAS_to_LPI = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        with open(folder_path + "/param.txt", "r") as f:
            text = f.read()
            vti_param = [p for p in text.split("\n") if p != ""]
            vti_venc = float(vti_param[2])
            vti_timesteps = int(vti_param[3])
            vti_size = np.array([int(num) for num in vti_param[4:]])

        with open(folder_path + "/spacing.txt", "r") as f:
            text = f.read()
            vti_spacing = np.array(
                [float(number) for number in text.split("\n") if number != ""]
            )
        with open(folder_path + "/offset.txt", "r") as f:
            text = f.read()
            vti_origin = np.array(
                [float(number) for number in text.split("\n") if number != ""]
            )
        with open(folder_path + "/matrix_poz.txt", "r") as f:
            text = f.read()
            matrix_poz = np.array(
                [
                    [float(number) for number in row.split(" ")]
                    for row in text.split("\n")
                    if row != ""
                ]
            )
        with open(folder_path + "/matrix_neg.txt", "r") as f:
            text = f.read()
            matrix_neg = np.array(
                [
                    [float(number) for number in row.split(" ")]
                    for row in text.split("\n")
                    if row != ""
                ]
            )
        spacing = 0.001 * vti_spacing
        timestep = 0.02  # TODO
        origin = 0.001 * np.dot(RAS_to_LPI, vti_origin)
        rotation = matrix_neg @ RAS_to_LPI
        other_parameters["venc"] = vti_venc
        return cls(
            size=vti_size,
            origin=origin,
            spacing=spacing,
            rotation=rotation,
            timestep=timestep,
            number_of_timesteps=vti_timesteps,
            data=data,
            **other_parameters,
        )

    @classmethod
    def from_json(cls, filename: str, **other_parameters):
        with open(filename + ".json", "r") as js:
            params = json.load(js)
        params["data"] = np.load(filename + ".npy")
        params["rotation"] = np.array(params["rotation"]).reshape(3, 3)
        return cls(
            **params,
            **other_parameters,
        )

    @property
    def size(self):
        return self.data.shape[2:5]

    @property
    def number_of_timesteps(self):
        return self.data.shape[0]

    @property
    def inv_rotation(self):
        return np.linalg.inv(self.rotation)

    @property
    def dim(self):
        return self.data.shape[1]

    def to_json(self, filename: str):
        params = {
            "size": [int(s) for s in self.size],
            "origin": [float(o) for o in self.origin],
            "spacing": [float(s) for s in self.spacing],
            "rotation": np.array(self.rotation).reshape(9).tolist(),
            "timestep": self.timestep,
            "number_of_timesteps": self.number_of_timesteps,
            "venc": self.venc,
        }
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder, exist_ok=True)
        pth = os.path.join(self.results_folder, filename)
        with open(pth + ".json", "w") as jsnfile:
            json.dump(params, jsnfile, indent=4)
        np.save(pth, self.data)
        return

    def to_vti(self, filename: str):
        "visualise data to vti (without rotation, origin and spacing considered)"
        data_type = vtk.VTK_FLOAT
        data = self.data
        shape = data.shape

        flat_data_array = data.flatten(order="F")
        vtk_data = numpy_support.numpy_to_vtk(
            num_array=flat_data_array, deep=True, array_type=data_type
        )

        img = vtk.vtkImageData()
        img.GetPointData().SetScalars(vtk_data)
        img.SetDimensions(shape[0], shape[1], shape[2])

        # Save the VTK file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(img)
        writer.Write()

    def xyz_to_ijk(self, xyz: np.ndarray) -> np.ndarray:
        """transform xyz from world to voxel coordinates"""
        img_coords = self.inv_rotation.dot(np.array(xyz) - self.origin)
        ijk = np.true_divide(img_coords, self.spacing)
        ijk_int = np.round(ijk).astype(int)
        return ijk_int

    def ijk_to_xyz(self, ijk: np.ndarray) -> np.ndarray:
        """transform ijk from voxel to world coordinates"""
        xyz = self.origin + self.rotation.dot(np.array(ijk) * self.spacing)
        return xyz

    @add_timing
    def dealiasing(
        self, data=None, axis: int = 0, direction: float = 1.0, threshold: float = 1.0
    ):
        """Dealias data based on axis, direction and threshold

        Args:
            data: data to dealias, take self.data if not provided
            axis: along which axis to dealias: 0->i, 1->j, 2->k
            direction: sign of this parameter determines direction of dealiasing
            threshold: how large should the velocity be to dealias
        """
        if data is None:
            data = np.copy(self.data)
        if abs(direction) != 1.0:
            direction = direction / abs(direction)
        if self.venc is None:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("Can't perform dealiasing because of missing VENC value.")
        else:
            data[:, axis][direction * data[:, axis] > threshold] -= (
                direction * 2.0 * self.venc
            )
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"Performed dealiasing on axis {axis} in direction {direction}.")
        return data

    @add_timing
    def apply_transformation(self, transformation: Optional[str]):
        if transformation is not None:
            # apply transformation obtained from registration in .mat format
            transform = sitk.ReadTransform(transformation)
            if transform.IsLinear():
                params = transform.GetParameters()
                rot_matrix = np.array(params[:9]).reshape(3, 3)
                translation = 0.001 * np.array(params[9:])
                print(
                    f"applying rotation and translation:\n {rot_matrix}\n {translation}"
                )
                self.origin = np.dot(rot_matrix, self.origin) + translation
                self.rotation = np.dot(rot_matrix, self.rotation)
            else:
                raise NotImplementedError("transform is nonlinear!")

    def resample_data_npy(
        self, npy_filenames: List[str], spacing: np.ndarray, scale=1.0
    ):
        self.spacing = spacing
        foldernames = [os.path.dirname(filename) for filename in npy_filenames]
        basenames = [os.path.basename(filename) for filename in npy_filenames]
        file_lists = [
            [f for f in os.listdir(foldername) if fnmatch(f, basename)]
            for basename, foldername in zip(basenames, foldernames)
        ]
        data_complete = []
        for j in range(len(file_lists)):
            file_lists[j].sort()
        for i in range(self.number_of_timesteps):
            directions = []
            for file_list, foldername in zip(file_lists, foldernames):
                arr = np.load(f"{foldername}/{file_list[i]}")
                directions.append(arr)
            if len(directions) == 3:
                directions[1] *= -1
                directions[2] *= -1
            data_complete.append(directions)
        data = np.array(data_complete)
        self.data = scale * data
        self.size = data.shape[-3:]

    @add_timing
    def apply_mask(
        self, mask_array: np.ndarray, data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply mask to the data and return the result as numpy array

        Args:
            data: numpy array containing the data to be masked
            mask: numpy array containing the mask
        """
        if data is None:
            data = self.data
        shape = data.shape
        masked_array = np.zeros(shape)
        mask = np.squeeze(mask_array)
        for i in range(shape[0]):
            for j in range(shape[1]):
                masked_array[i, j, :, :, :] = mask[:, :, :] * data[i, j, :, :, :]
        return masked_array


def trapezoidal_rule(times, values, t0, tN):
    f0 = values[0] + (values[1] - values[0]) * (t0 - times[0]) / (times[1] - times[0])
    fN = values[-2] + (values[-1] - values[-2]) * (tN - times[-2]) / (
        times[-1] - times[-2]
    )
    partial_sum = 0.5 * (f0 + values[1]) * (times[1] - t0) + 0.5 * (values[-2] + fN) * (
        tN - times[-2]
    )
    for j in range(1, len(times) - 2):
        partial_sum = partial_sum + 0.5 * (values[j + 1] + values[j]) * (
            times[j + 1] - times[j]
        )
    return partial_sum
