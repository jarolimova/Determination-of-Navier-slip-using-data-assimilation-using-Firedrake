"""This module provides utility classes for handling time steps, running averages, and data interpolation in simulations.
"""


class TimestepIterator:
    """Iterator for generating time steps based on a list of timestamps.
    The iterator yields tuples containing the timestamp, and the start and end of the interval around the timestamp.
    """

    def __init__(self, timestamps):
        """Initialize the iterator with a list of timestamps.

        Args:
            timestamps (list): A list of timestamps to iterate over.
        """
        timestamps.sort()
        self.timestamps = timestamps
        self.max_interval = max(
            [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        )

    def __iter__(self):
        self.iterator = iter(self.timestamps)
        return self

    def __next__(self):
        """Return the next timestamp and its interval."""
        t_next = next(self.iterator, None)
        if t_next is None:
            raise StopIteration
        return (
            t_next,
            t_next - self.max_interval / 2,
            t_next + self.max_interval / 2,
        )


class RunningAverage:
    """Class for computing a running average of a sequence of values.
    The average is updated with each new value provided.

    Args:
        zero: The initial value for the average - initializes its format.

    It can be float, fd.Function, or any other type that supports arithmetic operations.
    """

    def __init__(self, zero):
        self.current_average = zero
        self.number_of_averaged = 0

    def __call__(self, new_value):
        """Update the running average with a new value."""
        new_coef = 1.0 / (self.number_of_averaged + 1.0)
        avg_coef = self.number_of_averaged * new_coef
        self.current_average *= avg_coef
        self.current_average += new_coef * new_value
        self.number_of_averaged += 1


class DataInterpolation:
    """Class for interpolating data based on a dictionary of values at specific timesteps."""

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.timesteps = list(data_dict.keys())

    def __call__(self, t):
        """Interpolate the data at a given time t.
        If t is outside the range of the timesteps, it returns the closest available data.

        Args:
            t (float): The time at which to interpolate the data.

        Returns:
            Interpolated value at time t.
        """
        i = 0
        if t <= self.timesteps[0]:
            return self.data_dict[self.timesteps[0]]
        if t >= self.timesteps[-1]:
            return self.data_dict[self.timesteps[-1]]
        t_prev = self.timesteps[0]
        t_next = self.timesteps[1]
        while t >= t_next:
            i += 1
            t_prev = t_next
            t_next = self.timesteps[i]
        weight = (t - t_next) / (t_prev - t_next)
        interpolated = (1 - weight) * self.data_dict[t_next] + weight * self.data_dict[
            t_prev
        ]
        return interpolated


class SimulationTimestepping:
    """Class for managing simulation timesteps.
    It generates a sequence of timesteps based on an initial time and a fixed time step size.
    """

    def __init__(self, t0: float, dt: float):
        self.t0 = t0
        self.dt = dt

    def __call__(self, n: int):
        return self.t0 + (n + 1) * self.dt
