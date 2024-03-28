#!/usr/bin/env python
# Handy script for listing all available Gym environments | Praveen Palanisamy
# Chapter 4, Hands-on Intelligent Agents with OpenAI Gym, 2018

from gymnasium import envs
# print(f'envs.registry:{envs.registry.keys()}')
env_names = [spec for spec in envs.registry.keys()]
for name in sorted(env_names):
    print(name)
