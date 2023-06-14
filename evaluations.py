import os

from env_utils import UnevenMazeNormalized, UnevenMaze
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
tf.compat.v1.enable_eager_execution()

from uneven_maze.uneven_maze import sample_terrain_function
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

agent = Algorithm.from_checkpoint("checkpoints/checkpoint_30/")
# Test the trained agent
env = UnevenMazeNormalized(config)
s, _ = env.reset(options={"cost_step": 1.0, "cost_height": 50, "start_position": [0, 0]})
terminated = False
truncated = False
step_counter = 0
sum_r = 0
while not terminated and not truncated:
    # Changing the cost associated with uphill movement to zero after 10 steps :D
    if step_counter == 10:
        env.reset(options={"cost_step": 1.0, "cost_height": 0, "start_position": [int(s[2] * 10),
                                                                                   int(s[3] * 20)]})
    a = agent.compute_single_action(s, explore=False)
    s, r, terminated, truncated, info = env.step(a)
    print("step: ", step_counter, "s: ", s, "action: ", a, " reward: ", r, " terminated: ", terminated,
          " truncated: ", truncated)
    env.render()
    sum_r += r
    step_counter += 1
print(sum_r)
os.makedirs("images", exist_ok=True)
env._fig.savefig("images/episode_with_" + str(step_counter) +"_steps" + ".png")
