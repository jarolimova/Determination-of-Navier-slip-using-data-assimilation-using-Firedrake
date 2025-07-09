# Determination of inlet velocity and Navier slip using data assimilation by Firedrake

This repository contains the implementation of a data assimilation framework for estimating Navier’s slip boundary condition in cardiovascular flows. The method leverages PDE-constrained optimization informed by 4D phase-contrast MRI (4D flow MRI) data and is implemented using the [Firedrake](https://www.firedrakeproject.org/) finite element library.

The code supports both synthetic and real patient data and allows the estimation of spatially varying slip parameters and inflow velocity profiles in unsteady incompressible flow simulations.

## System requirements

The code has been developed and tested on **Ubuntu 20.04.6 LTS**. The following libraries are required:
- [Firedrake](https://www.firedrakeproject.org/)
- [Gmsh 4.11.1](https://gmsh.info/) - for mesh generation
- [vmtk 1.4.0](http://www.vmtk.org/) - for geometry processing

## Installation Guide

1. Install Firedrake
Follow the [official instructions](https://www.firedrakeproject.org/install.html) to install Firedrake.
   
2. Clone the repository and install the requirements.
   
        git clone https://github.com/jarolimova/Determination-of-Navier-slip-using-data-assimilation-using-Firedrake.git
        cd Determination-of-Navier-slip-using-data-assimilation-using-Firedrake
        pip install -r requirements.txt

The installation time is mostly dependent on the installation of Firedrake and should take less than an hour. 

## Demo

This demo showcases the complete pipeline for synthetic data generation and Navier slip parameter estimation using PDE-constrained optimization designed to work on a desktop computer.

### Step 1: Generate Morphological Image and Mesh

Simulate morphological imaging, generate a surface mesh, and build the volume mesh:

      python simulate_artificial_morphology.py tube_02 --voxelsize 0.002 --mesh_folder demo/data --vtk_folder demo/geometry --msh_paths demo/tube_02/tube_02
      python generate_surface_mesh.py demo/tube_02/mesh_preparation.json
      python generate_volume_mesh.py demo/tube_02/mesh_preparation.json

Estimated time: 5 minutes

Expected outputs:

   - `demo/geometry`: contains various files generated during the mesh generation process, the most important being `demo/geometry/tube_segmented/tube_segmented.msh` and the corresponding `_cuts.json` files

### Step 2: Generate Ground Truth Flow Data

This script simulates time-dependent artificial velocity fields for a predefined geometry (tube_02) with a prescribed slip value (θ = 0.8):

      python simulate_artificial_unsteady_data.py tube_02 0.8 --element p1p1 --name pulse --velocity 0.3 --velocity_factor pulse --startup none --meshpath demo --average_interval 0.05 --T 0.3 --dt 0.01 --data_folder demo/data

Estimated time: 2 minutes

Expected outputs:

   - `demo/data/pulse0.8_avg0.05_timedep.h5`: the velocity field downsampled in time
   - `demo/data/pulse0.8_avg0.05_timedep.json`: a list of timesteps corresponding to the velocity field saved with the `.h5` file of the same name
   - `demo/data/pulse0.8_timedep_pressure.h5`: the ground truth pressure field
   - `demo/data/pulse0.8_timedep.h5`: the ground truth velocity field
   - `demo/data/pulse0.8_timedep.json`: a list of timesteps corresponding to the timesteps saved with the `.h5` file of the same name
   - `demo/data/velocity_factor_pulse.png`: a picture showing the function used to vary the inlet velocity in time

### Step 3: Create Synthetic 4D Flow MRI

This script simulates MRI acquisition by downsampling and applying velocity encoding:

      python mri_artificial_data.py pulse0.8_avg0.05_timedep tube_02 --element p1p1 --venc 1.0 --voxelsize 0.003 --unsteady --data_folder demo/data --MRI_folder demo/MRI_npy --msh_paths demo/geometry/tube_segmented/tube_segmented

Estimated time: 2 minutes

Expected outputs: 

   - `demo/MRI_npy/pulse0.8_tube_3.0mm.json`: header containing all the necessary information about the MRI object
   - `demo/MRI_npy/pulse0.8_tube_3.0mm.npy`: data corresponding to the header of the same name
   - two other pairs of `.json` + `.npy` files including data with added noise with SNR = 5 and SNR = 3
   - `demo/data/tube_02.h5`: mesh resaved to `h5` format
   - `demo/data/tube_02_cuts.json`: `JSON` file containing information about the inlet and outlet boundaries

### Step 4: Assimilate and Estimate Slip Parameter

Perform PDE-constrained optimization using synthetic 4D flow MRI to recover both the inflow profile and the optimal Navier slip parameter:

      python unsteady_assimilation.py tube_segmented pulse0.8_tube_snr5_3.0mm --alpha 1e-3 --gamma 1e-3 --epsilon 1e-3 --element p1p1 --init_theta 0.7 --dt 0.02 --vin_path data --MRI_space CG --presteps 2 --ftol 1e-5 --gtol 1e-4 --average --stabilization IP --MRI_json_folder demo/MRI_npy --data_folder demo/data --results_folder demo/results

Estimated time: 20 minutes

Expected outputs are all located in folder `demo/results/tube_segmented/pulse0.8_tube_snr5_3.0mm_CG/p1p1_stab0.0005_1.0_0.01/init_theta0.7_data/facet_0.25_T0.25_dt0.02_pr2/alpha0.001_gamma0.001_eps0.001_avg/picard0` and contain the following:
 
   - `data.pvd`: visualization file of the MRI data interpolated to the computational mesh
   - `mri.pvd`: visualization file of the MRI data
   - `ns_opt_data_est.pvd`: visualization file of optimal velocity and pressure field downsampled in time
   - `ns_opt.pvd`: visualization file of optimal velocity and pressure field
   - `ns_opt.h5`: optimal velocity field in HDF5 format
   - `p_opt.h5`: optimal pressure field in HDF5 format
   - `u_in_opt.h5`: optimal inflow velocity profile in HDF5 format
   - `u_in_start.pvd`: initial guess for the inflow velocity profile
   - `uin_chpt.h5`: checkpoint file for inflow velocity during optimization
   - `output.csv`: summary of optimization results 

### Step 5: Plot and Visualize Results

All the files in `demo/results/tube_segmented/pulse0.8_tube_snr5_3.0mm_CG/p1p1_stab0.0005_1.0_0.01/init_theta0.7_data/facet_0.25_T0.25_dt0.02_pr2/alpha0.001_gamma0.001_eps0.001_avg/picard0` folder with in `.pvd` format can be viewed in [Paraview](https://www.paraview.org/).
Other option to visualize the results is to run postprocessing scripts:

      python postprocessing_pdrop_vavg.py --meshes tube_02 --datanames pulse0.8_tube_snr5_3.0mm_CG --MRI_data demo/MRI_npy/pulse0.8_tube_snr5_3.0mm --ground_truth_h5 demo/data/tube_02/p1p1/pulse0.8_timedep --labels slip --results_folder demo/results --data_folder demo/data --plot_folder demo/plots
      python postprocessing_velocity_on_wall.py tube_02 pulse0.8_tube_snr5_3.0mm_CG --discretization facet_0.25_T0.25_dt0.02_pr2 --results_folder demo/results --plot_folder demo/plots 

Estimated time: 2 minutes

Expected outputs:
      - `demo/plots/pressure_drop_tube_segmented_facet_0.25_T0.25_dt0.02` - plot showing the change of average velocity and pressure drop over time
      - `demo/plots/v_on_wall_tube_segmented` - plot comparing the average velocity on wall versus the average velocity in bulk over time
      - `demo/plots/relative_tube_segmented` - plot of the ratio of velocity on wall and in bulk over time

## Instructions for use

To apply the method to any other data, artificial or real, follow these general steps. To get more information about the parameters provided to each of the scripts, run it with `--help` switch to obtain more information.

### Artificial Data

The artificial data can be generated the same way as shown in the demo. 

1. The artificial morphology image is created using `simulate_artificial_morphology.py` and computational mesh is created using `generate_surface_mesh.py` and `generate_volume_mesh.py` by providing the corresponding json file.
2. The ground truth velocity and pressure field are computed and saved using `simulate_artificial_unsteady_data.py`
3. The artificial MRI data are created using `mri_artificial_data.py`
4. The assimilation is run using `unsteady_assimilation.py`

### Real Patient Data

1. The geometry has to be first segmented using some software such as [ITK-SNAP](https://www.itksnap.org/pmwiki/pmwiki.php) or [3D Slicer](https://www.slicer.org/) and saved in a `.mha` or `.vtk` format (other formats might work as well but were not tested).
2. The computational mesh can be then generated using `generate_surface_mesh.py` and `generate_volume_mesh.py` by providing the corresponding json file.
3. The MRI flow data are read from the source files using the script `mri_volunteers.py`
4. The assimilation is run using `using_assimilation.py`
