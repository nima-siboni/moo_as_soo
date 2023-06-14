from uneven_maze.uneven_maze import UnevenMaze

# create a wrapper around UnevenMaze where the observation is normalized.
# This is useful for training neural networks.

import gymnasium as gym
import numpy as np


class UnevenMazeNormalized(UnevenMaze):
    def __init__(self, config):
        super().__init__(config)

        # define the observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )

    def _get_observation(self):
        """
        Get the observation of the current state.
        :return: the observation
        """
        # get the observation from the parent class
        observation = super()._get_observation()

        # normalize the observation
        cost_height_max = self._cost_height_max if self._cost_height_max > 0 else 1
        cost_step_max = self._cost_step_max if self._cost_step_max > 0 else 1
        observation = observation / np.array(
            [cost_height_max, cost_step_max, self.height + 1, self.width + 1]
        )

        return observation

