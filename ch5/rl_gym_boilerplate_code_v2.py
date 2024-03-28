#!/usr/bin/env python
# Boilerplate code for Reinforcement Learning with Gym | Praveen Palanisamy
# Chapter 5, Hands-on Intelligent Agents with OpenAI Gym, 2018

import gymnasium as gym
env = gym.make("MountainCar-v0", render_mode="human")
MAX_NUM_EPISODES = 1

for episode in range(MAX_NUM_EPISODES):
    terminated = False
    obs = env.reset()
    total_reward = 0.0 # To keep track of the total reward obtained in each episode
    step = 0
    while not terminated or not truncated:
        env.render()
        action = env.action_space.sample()# Sample random action. This will be replaced by our agent's action when we start developing the agent algorithms
        next_state, reward, terminated, truncated, info = env.step(action) # Send the action to the environment and receive the next_state, reward and whether done or not
        total_reward += reward
        step += 1
        obs = next_state
        if step > 5000:
            break

    print("\n Episode #{} ended in {} steps. total_reward={}".format(episode, step+1, total_reward))
env.close()
