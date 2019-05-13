#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt

from agents.ddpg_agent import DDPGAgent
from task import Task


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def main():
    num_episodes = 100
    init_pose = np.array([0., 0., 30., 0., 0., 0.])
    init_velocities = np.array([0., 0., 10.])
    target_pos = np.array([0., 0., 40.])
    max_time_s = 5.0
    task = Task(init_pose=init_pose, init_velocities=init_velocities, target_pos=target_pos, runtime=max_time_s)
    agent = DDPGAgent(task)

    labels = ['episode', 'reward', 'final_pose']
    results = {x: [] for x in labels}

    count_success = 0
    count_failure = 0

    for i_episode in range(1, num_episodes+1):
        print('=== Beginning episode #' + str(i_episode))
        state = agent.reset_episode()  # start a new episode
        total_episode_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            total_episode_reward += reward
            agent.step(action, reward, next_state, done)
            state = next_state
            if done:
                results['episode'].append(i_episode)
                results['reward'].append(total_episode_reward)
                results['final_pose'].append(task.sim.pose)
                print('Final step:', task.sim.time, task.sim.pose[:3], task.sim.v, reward)
                # print("Episode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                #     i_episode, agent.last_score, agent.best_score, agent.noise_scale))  # [debug]
                print()
                if task.target_reached:
                    count_success += 1
                else:
                    count_failure += 1
                break
        sys.stdout.flush()

    print('Successes:', count_success, '- Failures:', count_failure)

    plt.plot(results['episode'], results['reward'], label='reward')
    smoothed_rews = running_mean(results['reward'], 50)
    plt.plot(results['episode'][-len(smoothed_rews):], smoothed_rews, label='average_reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
