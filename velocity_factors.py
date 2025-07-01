from abc import ABC, abstractmethod
import numpy as np
from typing import List
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev, splrep

__all__ = [
    "VelocityFactor",
    "ConstVelocityFactor",
    "TransientVelocityFactor",
    "PulseVelocityFactor",
]


class VelocityFactor(ABC):
    """Abstract base class for velocity factor"""

    @abstractmethod
    def __call__(self, t: float) -> float:
        pass


class ConstVelocityFactor(VelocityFactor):
    """Velocity factor used for steady flow (with constant velocity)"""

    def __init__(self, velocity_average: float = 1.0):
        """Initialization of the object

        args:
            velocity_average: The velocity magnitude averaged over the inlet, default=1.0
        """
        self.v = velocity_average

    def __call__(self, t: float) -> float:
        """Return the velocity factor at the given time t

        args:
            t: time stamp
        """
        return self.v


class TransientVelocityFactor(VelocityFactor):
    """Velocity factor used for transient periodic flow defined using cubic splines

    if the fist and last velocity values are not the same, append the first value at the end with the timestep size the same between the first two values
    """

    def __init__(
        self,
        timestamps: List[float] = [],
        velocities: List[float] = [],
        smoothing: float = 0.001,
    ):
        """Initialization of the object

        args:
            timestamps: list of timestamps
            velocities: list of corresponding velocities
            smoothing: amount of smoothing (0.0 mean no smoothing), default=0.001
        """
        self.timestamps = timestamps
        self.velocities = velocities
        if velocities[0] != velocities[-1]:
            velocities.append(velocities[0])
            timestamps.append(timestamps[-1] + timestamps[1] - timestamps[0])
        spl = splrep(timestamps, velocities, s=smoothing, per=True)
        smdata = splev(timestamps, spl)
        self.v = CubicSpline(timestamps, smdata, bc_type="periodic")

    @classmethod
    def from_csv(cls, csv_path: str):
        """Alternative initialization of the object from a csv file

        args:
            csv: path to a csv file with the timestamps in the first column and velocities in the second column
        """
        inflow_data = np.genfromtxt(csv_path, unpack=True)
        return cls(inflow_data[0], inflow_data[1])

    def __call__(self, t: float) -> float:
        """Return the velocity factor at the given time t

        args: t: time stamp
        """
        return self.v(t)


class PulseVelocityFactor(VelocityFactor):
    """A synthetic pulse which can be used for transient periodic flow
    The default setting was used for article cibule
    """

    def __init__(self, period: float = 1.0, t_max: float = 0.15, v_max: float = 1.0):
        """Initialization of the object

        args:
            period: length of one pulse
            t_max: time in the first period when the maximal velocity is attained
            v_max: the maximal velocity
        """
        self.period = period
        self.t_max = t_max
        self.v_max = v_max
        self.coef = v_max / pow(t_max, 2)

    def __call__(self, t: float) -> float:
        """Return the velocity factor at the given time t

        args: t: time stamp
        """
        base_t = t % self.period
        if base_t < 2 * self.t_max:
            return self.v_max - self.coef * pow(base_t - self.t_max, 2)
        else:
            return 0.0


string_to_velocity_factor = {
    "const": ConstVelocityFactor,
    "transient": TransientVelocityFactor,
    "pulse": PulseVelocityFactor,
}


def plot_velocity_factor(
    filepath: str, velocity_factor: VelocityFactor, start: float = 0.0, end: float = 1.0
):
    """Plot a given velocity factor and save it to filepath"""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    figure(figsize=(6.4, 2.4), dpi=80)
    time = np.arange(start, end, (end - start) / 100)
    vals = [velocity_factor(t) for t in time]

    plt.plot(time, vals, "k--", label="velocity factor")
    try:
        plt.plot(
            velocity_factor.timestamps, velocity_factor.velocities, "ko", label="data"
        )
    except:
        pass
    plt.xlabel("time [s]")
    plt.ylabel("avg velocity [m/s]")
    plt.legend(loc="upper right")
    plt.savefig(f"{filepath}.png", bbox_inches="tight")
    plt.clf()
    return
