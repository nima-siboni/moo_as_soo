import os

import matplotlib.pyplot as plt
import numpy as np
from env_utils import UnevenMazeNormalized
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
tf.compat.v1.enable_eager_execution()

from uneven_maze import sample_terrain_function
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

config = {
    "width": 20,
    "height": 10,
    "mountain_height": 1.0,
    "goal_position": [10, 0],
    "max_steps": 100,
    "cost_height_max": 50.0,
    "cost_step_max": 1.0,
    "cost_step_min": 1.0,
    "terrain_function": sample_terrain_function,
}

# Register the environment
register_env("uneven_maze", lambda conf: UnevenMazeNormalized(conf))

agent = Algorithm.from_checkpoint("checkpoints/checkpoint_50/")
# Test the trained agent
env = UnevenMazeNormalized(config)

q_values = np.zeros(shape=(config["height"], config["width"]))

for i in range(config["height"]):
    for j in range(config["width"]):
        s, _ = env.reset(
            options={"cost_step": 1.0, "cost_height": 50, "start_position": [i, j]}
        )
        q = agent.compute_single_action(s, explore=False, full_fetch=True)[2][
            "q_values"
        ]
        q_max = np.max(q)
        q_values[i, j] = q_max
plt.imshow(q_values)
plt.savefig("images/learned_values_w_cost_height_" + str(env.cost_height) + ".png")
print("bye")
