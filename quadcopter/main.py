#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt

from agents.agent import DDPGAgent
from task import Task


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


num_episodes = 500
init_pose = np.array([0., 0., 25., 0., 0., 0.])
init_velocities = np.array([0., 0., 15.])
target_pos = np.array([0., 0., 50.])
max_time_s = 5.0
task = Task(init_pose=init_pose, init_velocities=init_velocities, target_pos=target_pos, runtime=max_time_s)
agent = DDPGAgent(task)

labels = ['episode', 'reward', 'final_pose']
results = {x: [] for x in labels}

count_success = 0
count_failure = 0
count_success_sequence = 0

best_episode_reward = -np.inf
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode()  # start a new episode
    total_episode_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        total_episode_reward += reward
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            if total_episode_reward > best_episode_reward:
                best_episode_reward = total_episode_reward
            results['episode'].append(i_episode)
            results['reward'].append(total_episode_reward)
            results['final_pose'].append(task.sim.pose)
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                  i_episode, total_episode_reward, best_episode_reward, 0), end="")  # [debug]
            if task.target_reached:
                count_success += 1
                count_success_sequence += 1
            else:
                count_failure += 1
                count_success_sequence = 0
            break
    sys.stdout.flush()
    if count_success_sequence > 10:
        break

print()
print('Successes:', count_success, '- Failures:', count_failure)

plt.plot(results['episode'], results['reward'], label='rewards')
plt.legend()
plt.show()
