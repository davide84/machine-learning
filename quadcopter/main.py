#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd

from agents.agent import MyAgent
from task import Task


def main():
    num_episodes = 1000
    init_pose = np.array([0., 0., 10., 0., 0., 0.])
    target_pos = np.array([10., 10., 10.])
    task = Task(init_pose=init_pose, target_pos=target_pos)
    agent = MyAgent(task)

    labels = ['episode', 'reward', 'final_pose']
    results = {x: [] for x in labels}

    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(reward, done)
            state = next_state
            if done:
                results['episode'].append(i_episode)
                results['reward'].append(agent.last_score)
                results['final_pose'].append(task.sim.pose)
                print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                    i_episode, agent.last_score, agent.best_score, agent.noise_scale), end="")  # [debug]
                break
        sys.stdout.flush()


if __name__ == '__main__':
    main()
