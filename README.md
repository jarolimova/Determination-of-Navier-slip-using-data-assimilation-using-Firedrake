# Determination of Navier slip using data assimilation using Firedrake

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

## Demo

This demo showcases the complete pipeline for synthetic data generation and Navier slip parameter estimation using PDE-constrained optimization designed to work on a desktop computer.

### Step 1: Generate Ground Truth Flow Data

This script simulates time-dependent artificial velocity fields for a predefined geometry (tube_02) with a prescribed slip value (θ = 0.8):

      python simulate_artificial_unsteady_data.py tube_02 0.8 --element p1p1 --name pulse --velocity 0.3 --velocity_factor pulse --startup none --meshpath demo --average_interval 0.05 --T 0.3 --dt 0.01 --data_folder demo/data

### Step 2: Create Synthetic 4D Flow MRI

This script simulates MRI acquisition by downsampling and applying velocity encoding:

      python mri_artificial_data.py pulse0.8_avg0.05_timedep tube_02 --element p1p1 --venc 1.0 --voxelsize 0.003 --unsteady --data_folder demo/data --MRI_folder demo/MRI_npy --msh_paths demo/tube_02/tube_02

### Step 3: Generate Morphological Image and Mesh

Simulate morphological imaging, generate a surface mesh, and build the volume mesh:

      python simulate_artificial_morphology.py tube_02 --voxelsize 0.002 --mesh_folder demo/data --vtk_folder demo/geometry
      python3 generate_surface_mesh.py demo/tube_02/mesh_preparation.json
      python3 generate_volume_mesh.py demo/tube_02/mesh_preparation.json

### Step 4: Assimilate and Estimate Slip Parameter

Perform PDE-constrained optimization using synthetic 4D flow MRI to recover both the inflow profile and the optimal Navier slip parameter:

      python unsteady_assimilation.py tube_02 pulse0.8_tube_snr5_3.0mm --alpha 1e-3 --gamma 1e-3 --epsilon 1e-3 --element p1p1 --init_theta 0.7 --dt 0.02 --vin_path data --MRI_space CG --presteps 2 --ftol 1e-5 --gtol 1e-4 --average --stabilization IP --MRI_json_folder demo/MRI_npy --data_folder demo/data --results_folder demo/results

## Instructions for use

