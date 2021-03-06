""" Implement a trajectory controller on top of a controller

Implements a second order filter for path generation.
Returns a set of target positions and velocities.
Implements the second order filter from
www.mathworks.com/help/physmod/sps/powersys/ref/secondorderfilter.html

returns target in form [positions, velocities]

***NOTE*** there are three ways to use this filter
1: wrt to timesteps (step at a time)
- each step (from 0 to n_timesteps) of the path planner is generated by the step
function.

2: wrt to timesteps (pregenerated)
- the step() function can be called in a loop, to pregenerate the path, by
calling the generate_path() function. Then each step can be passed sequentially
by calling the next_target() function in a loop

3: wrt to time
- after instantiation, calling `generate_path_function()` interpolates the path
to the specified time limit. Calling the `next_timestep(t)` function at a
specified time will return the end-effector state at that point along the path
planner. This ensures that the path will reach the desired target within the
time_limit specified in `generate_path_function()`
"""
import numpy as np

from .path_planner import PathPlanner


class SecondOrder(PathPlanner):
    """
    Parameters
    ----------
    n_timesteps: int, optional (Default: 3000 ~3sec given a 3ms comm. loop)
        the number of time steps to reach the target
    dt: float, optional (Default: 0.001)
        the loop speed [seconds]
    zeta: float, optional (Default: 2.0)
        the damping ratio
    w: float, optional (Default: 1e-4)
        the natural frequency
    threshold: float, optional (Default: 0.02)
        within this threshold distance to target position reduce the
        filtering effects to improve convergence in practice
    """
    def __init__(self, n_timesteps=3000, dt=0.001,
                 zeta=2.0, w=1e4, threshold=0.02):

        self.n_timesteps = n_timesteps
        self.dt = dt
        self.zeta = zeta
        self.w = w/n_timesteps # gain to converge in the desired time
        self.threshold = threshold


    def generate_path(self, position, velocity, target_pos, plot=False):
        """
        Calls the step function self.n_timestep times to pregenerate
        the entire path planner

        Parameters
        ----------
        position: numpy.array
            the current position of the system
        velocity: numpy.array
            the current velocity of the system
        target_pos: numpy.array
            the target position
        plot: boolean, optional (Default: False)
            plot the path after generating if True
        """

        position_path = []
        velocity_path = []

        for _ in range(self.n_timesteps):
            position_path.append(position)
            velocity_path.append(velocity)
            position, velocity = self._step(position, velocity, target_pos)

        self.position = np.array(position_path)
        self.velocity = np.array(velocity_path)

        # reset trajectory index
        self.n = 0

        if plot:
            self.plot(target_pos)

        return self.position, self.velocity


    def _step(self, position, velocity, target_pos):
        """ Calculates the next state given the current state and
        system dynamics' parameters.

        Parameters
        ----------
        position: numpy.array
            the current position of the system
        velocity: numpy.array
            the current velocity of the system
        target_pos: numpy.array
            the target position of the system
        """

        w = self.w
        if np.linalg.norm(position - target_pos) < self.threshold:
            # if within a threshold distance, reduce the filter effect
            # NOTE: this is a ad-hoc method of improving performance at
            # short distances
            w *= 3

        accel = (w**2 * target_pos
                 - velocity * self.zeta * w
                 - position * w**2)
        velocity = velocity + accel * self.dt
        position = position + velocity * self.dt

        return position, velocity
