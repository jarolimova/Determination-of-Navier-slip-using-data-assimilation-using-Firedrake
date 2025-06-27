""" SlipPenalty class for computing slip penalty on the wall in fluid dynamics simulations.
"""

from math import log, e

NU = 3.71e-6  # kinematic viscosity
RHO = 1050.0  # density
CHAR_LEN = 0.01  # approx average radius


class SlipPenalty:
    """Object created for computation of slip penalty on the wall
    defined using theta, penalty = theta/(gamma*(1-theta)), kappa = 1/penalty, logarithm = log(penalty)
    theta = gamma*penalty/(1+ gamma*penalty)
    Args:
        control_variable: optimized quantity, options: theta, penalty, kappa, logarithm
        gamma: gamma parameter in the wall bc
    """

    def __init__(
        self,
        control_variable: str = "theta",
        gamma: float = 1.0,
        mu: float = 1050 * 3.71 * 1e-6,
    ):
        self.gamma = gamma
        self.mu = mu
        self.control_variable = control_variable
        if control_variable == "theta":
            self.control_variable_to_penalty = self.theta_to_penalty
            self.penalty_to_control_variable = self.penalty_to_theta
        elif control_variable == "logarithm":
            self.control_variable_to_penalty = self.logarithm_to_penalty
            self.penalty_to_control_variable = self.penalty_to_logarithm
        elif control_variable == "penalty":
            self.control_variable_to_penalty = lambda x: x
            self.penalty_to_control_variable = lambda x: x
        elif control_variable == "sliplength":
            self.control_variable_to_penalty = self.penalty_to_sliplength_and_back
            self.penalty_to_control_variable = self.penalty_to_sliplength_and_back
        else:
            raise ValueError(
                f"invalid control variable {control_variable}, choose one of the following options instead: theta, penalty, logarithm"
            )

    def theta_to_penalty(self, theta):
        """compute penalty using theta"""
        return theta / (self.gamma * (1.0 - theta))

    def penalty_to_theta(self, penalty):
        """compute theta using penalty"""
        return self.gamma * penalty / (1 + self.gamma * penalty)

    def penalty_to_logarithm(self, penalty):
        """compute logarithm using penalty"""
        return log(penalty)

    def logarithm_to_penalty(self, logarithm):
        """compute penalty using logarithm"""
        return e**logarithm

    def penalty_to_sliplength_and_back(self, penalty):
        """compute slip length using penalty"""
        return self.mu / penalty

    def control_variable_to_theta(self, control_variable):
        """compute theta using control variable defined by self.control_variable string"""
        penalty = self.control_variable_to_penalty(control_variable)
        return self.penalty_to_theta(penalty)

    def theta_to_control_variable(self, theta):
        """compute control variable defined by self.control_variable string using theta"""
        penalty = self.theta_to_penalty(theta)
        return self.penalty_to_control_variable(penalty)


if __name__ == "__main__":
    """
    Print theta, penalty and slip length for different volunteers and synthetic meshes
    """
    import os
    import pandas as pd

    slip_penalty = SlipPenalty(gamma=0.25)
    results_folder = "/usr/work/jarolimova/assimilation/results_firedrake"
    element_inits = "p1p1_stab0.0005_1.0_0.01/init_theta0.7_nitsche_data"
    reg_solver = "alpha0.001_gamma0.001_eps0.001_avg/picard0"

    volunteers = ["vol01", "vol03", "vol10", "vol13", "vol17", "vol19", "vol20"]
    for vol in volunteers:
        meshes = [
            f"long_015_{vol}",
            f"segmentator_015_{vol}",
            f"segmentator_wider_015_{vol}",
        ]
        for meshname in meshes:
            dataname = f"{vol}_CG"
            print(meshname)
            path = os.path.join(results_folder, meshname, dataname, element_inits)
            discretization = [
                disc for disc in os.listdir(path) if "dt0.01" in disc and "pr2" in disc
            ][0]
            path = os.path.join(path, discretization, reg_solver)
            csv_data = pd.read_csv(os.path.join(path, "output.csv"))
            theta = csv_data["theta"].tolist()[-1]
            penalty = slip_penalty.theta_to_penalty(theta)
            slip_length = NU * RHO / penalty
            print(
                f"theta = {round(theta, 4)}, penalty = {round(penalty,4)}, slip_length = {round(1000*slip_length, 4)} mm"
            )
        print("")
    """
    synthetic_meshes = ["tube_seg01_015", "bent_aorta_seg01_015", "arch_seg01_015"]
    for meshname in synthetic_meshes:
        short_meshname = meshname.split("_")[0]
        print(meshname)
        for theta in [0.5, 0.999]:
            print(theta)
            dataname = f"pulse{theta}_{short_meshname}_proj_CG"
            path = os.path.join(results_folder, meshname, dataname, element_inits)
            discretization = os.listdir(path)[0]
            path = os.path.join(path, discretization, reg_solver)
            csv_data = pd.read_csv(os.path.join(path, "output.csv"))
            theta = csv_data["theta"].tolist()[-1]
            penalty = slip_penalty.theta_to_penalty(theta)
            slip_length = NU * RHO / penalty
            print(
                f"theta = {round(theta, 4)}, penalty = {round(penalty,4)}, slip_length = {round(1000*slip_length, 4)} mm"
            )
            print("")
    """
