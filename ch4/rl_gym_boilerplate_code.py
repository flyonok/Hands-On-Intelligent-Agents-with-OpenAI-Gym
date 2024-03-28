#!/usr/bin/env python
# Boilerplate code for Reinforcement Learning with Gym | Praveen Palanisamy
# Chapter 4, Hands-on Intelligent Agents with OpenAI Gym, 2018

import gymnasium as gym
# env = gym.make("ALE/Qbert-v5", render_mode="human")
env = gym.make("ALE/Boxing-v5", render_mode="human")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
for episode in range(MAX_NUM_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()# Sample random action. This will be replaced by our agent's action when we start developing the agent algorithms
        next_state, reward, terminated, truncated, info = env.step(action) # Send the action to the environment and receive the next_state, reward and whether done or not
        obs = next_state

        if terminated is True:
            print("\n Episode #{} ended in {} steps.".format(episode, step+1))
            break
        
