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



## Instructions for use

