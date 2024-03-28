#!/usr/bin/env python
# Handy script for exploring the available Gym environments | Praveen Palanisamy
# Chapter 4, Hands-on Intelligent Agents with OpenAI Gym, 2018

import gymnasium as gym
import sys

def run_gym_env(argv):
    env = gym.make(argv[1], render_mode="human") # Name of the environment supplied as 1st argument
    env.reset()
    for _ in range(int(argv[2])): # Number of steps to be run supplied as 2nd argument
        env.render()
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f'reward:{reward}, observation:{observation}')
        # if terminated or truncated:
        if terminated:
            break
    env.close()
    
if __name__ == "__main__":
    run_gym_env(sys.argv)
