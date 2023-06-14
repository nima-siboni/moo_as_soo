import os

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from env_utils import UnevenMazeNormalized
from uneven_maze.uneven_maze import sample_terrain_function
from ray.tune.registry import register_env

config = {
    "width": 20,
    "height": 10,
    "mountain_height": 1.0,
    "goal_position": [10, 0],
    "max_steps": 100,
    "cost_height_max": 50.0,
    "cost_height_min": 0.0,
    "cost_step_max": 1.0,
    "cost_step_min": 0.0,
    "terrain_function": sample_terrain_function,
}

# Register the environment
register_env("uneven_maze", lambda conf: UnevenMazeNormalized(conf))

# Define the configuration of the DQN algorithm
agent = DQNConfig().\
    framework("tf2").\
    environment(env="uneven_maze", env_config=config).\
    rollouts(num_envs_per_worker=4, num_rollout_workers=4).\
    build()

for _ in range(200):
    history = agent.train()
    print(_, history["episode_reward_mean"])
    # create a directory to save the checkpoints every 10 iterations
    if _ % 10 == 0:
        os.makedirs("checkpoints/checkpoints_"+str(_), exist_ok=True)
        agent.save_checkpoint("checkpoints/checkpoints_"+str(_))
