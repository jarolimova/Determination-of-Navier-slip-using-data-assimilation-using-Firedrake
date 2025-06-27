"""Startups can be used in inflows for FEM simulations to damp the velocity profile at the start of the simulation.
string_to_startup dict gives a mapping between strings and startups to avoid calling getattr: startup = string_to_startup["startup_name"](start=0.0, end=1.0)
To see the difference between all implemented startups, use the plot_startups function: plot_startups(plotpath, start=0.0, end=1.0)
"""

import math

__all__ = [
    "DefaultStartup",
    "LinearStartup",
    "CosineStartup",
    "ExponentialStartup",
    "string_to_startup",
    "plot_startups",
]


class DefaultStartup:
    """A base class for startup, which does not change the inflow profile. (Returns 1.0 for every value of time.)

    Example:
        def_startup = DefaultStartup()
        def_startup(t=0.5)
    """

    def __init__(self, start: float = 0.0, end: float = 1.0):
        self.start = start
        self.end = end

    def __call__(self, t: float):
        """retrieve the startup value in the actual time instant"""
        return 1.0


class LinearStartup(DefaultStartup):
    """A linear startup for velocity inflow profiles.

    Args:
        start: start of the startup - for lower values it is indentical zero
        end: end of the startup - the startup does not affect the profile for higher values of t (it is equal to 1)

    Example:
        def_startup = LinearStartup(start=0.0, end=4.0)
        def_startup(t=2.0)
    """

    def __call__(self, t: float):
        if t >= self.end:
            return 1.0
        elif t > self.start:
            interval = self.end - self.start
            duration = t - self.start
            return duration / interval
        else:
            return 0.0


class CosineStartup(DefaultStartup):
    """A cosine startup for velocity inflow profiles.

    Args:
        start: start of the startup - for lower values it is indentical zero
        end: end of the startup - the startup does not affect the profile for higher values of t (it is equal to 1)

    Example:
        def_startup = CosineStartup(start=0.0, end=4.0)
        def_startup(t=2.0)
    """

    def __call__(self, t: float):
        if t >= self.end:
            return 1.0
        elif t > self.start:
            interval = self.end - self.start
            duration = t - self.start
            return 0.5 * (1.0 - math.cos(math.pi * duration / interval))
        else:
            return 0.0


class ExponentialStartup(DefaultStartup):
    """An exponential startup for velocity inflow profiles.
    The value of this startup does not reach 1.0 but it is getting closer as t approaches infinity

    Args:
        start: start of the startup - for lower values it is indentical zero
        end: end of the startup

    Example:
        def_startup = ExponentialStartup(start=0.0, end=4.0)
        def_startup(t=2.0)
    """

    def __call__(self, t: float):
        if t > self.start:
            interval = self.end - self.start
            duration = t - self.start
            return 1.0 - math.exp(-2.5 * duration / interval)
        else:
            return 0.0


string_to_startup = {
    "none": DefaultStartup,
    "linear": LinearStartup,
    "cosine": CosineStartup,
    "exponential": ExponentialStartup,
}


def plot_startups(filepath: str, start: float = 0.0, end: float = 1.0):
    """Plot startups and save to a given location

    Args:
        filepath: path to the location where the plot should be saved (without extension)
        start: start of the plotted startups
        end: end of the plotted startups
    """
    import numpy as np
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    arg = {"start": start, "end": end}

    a, b = (arg["start"] - 0.5, arg["end"] + 0.5)
    time = np.arange(a, b, (b - a) / 50)

    non = DefaultStartup()
    lin = LinearStartup(**arg)
    cos = CosineStartup(**arg)
    exp = ExponentialStartup(**arg)

    non_vals = []
    lin_vals = []
    cos_vals = []
    exp_vals = []

    for t in time:
        non_vals.append(non(t))
        lin_vals.append(lin(t))
        cos_vals.append(cos(t))
        exp_vals.append(exp(t))

    plt.plot(time, non_vals, "r--", label="none")
    plt.plot(time, lin_vals, "g--", label="linear")
    plt.plot(time, cos_vals, "b--", label="cosine")
    plt.plot(time, exp_vals, "c--", label="exponential")

    plt.xlabel("time [s]")
    plt.savefig(f"{filepath}.png", bbox_inches="tight")
