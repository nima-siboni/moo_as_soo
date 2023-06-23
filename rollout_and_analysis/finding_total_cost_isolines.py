import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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

agent = Algorithm.from_checkpoint("checkpoints/checkpoint_490/")
# Test the trained agent
env = UnevenMazeNormalized(config)

# an array for cost hight starting from 0 to 50 with step 1
# an array for cost step starting from 0.65 to 1 with step 0.02

cost_height_range = np.arange(0, 51, 1)
cost_step_range = np.arange(0.5, 1.01, 0.01)

pareto_front = []
for cost_height in tqdm(cost_height_range):
    for cost_step in cost_step_range:
        s, _ = env.reset(
            options={
                "cost_step": cost_step,
                "cost_height": cost_height,
                "start_position": [0, 0],
            }
        )
        terminated = False
        truncated = False
        sum_cost_step = 0
        sum_cost_height = 0
        while not (terminated or truncated):
            a = agent.compute_single_action(s, explore=False)
            s, r, terminated, truncated, info = env.step(a)
            sum_cost_step += -1 * info["cost_step"]
            sum_cost_height += -1 * info["cost_height"]
        pareto_front.append([cost_height, cost_step, sum_cost_step, sum_cost_height])

pareto_front = np.array(pareto_front)
np.save("cost_height_step_total_step_total_height.npy", pareto_front)

# Plot the contour plot with x-axis as cost step and y-axis as cost height
# and the color as the time
plt.figure()
plt.xlabel("cost step")
plt.ylabel("cost height")
plt.contourf(
    cost_step_range,
    cost_height_range,
    (pareto_front[:, 2] + pareto_front[:, 3]).reshape(
        len(cost_height_range), len(cost_step_range)
    ),
    levels=20,
)
plt.colorbar()
plt.savefig("images/total_cost_contour.png")
print("bye")
