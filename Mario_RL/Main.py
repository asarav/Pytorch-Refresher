# -*- coding: utf-8 -*-
"""
Train a Mario-playing RL Agent
================

Authors: `Yuansong Feng <https://github.com/YuansongFeng>`__, `Suraj
Subramanian <https://github.com/suraj813>`__, `Howard
Wang <https://github.com/hw26>`__, `Steven
Guo <https://github.com/GuoYuzhang>`__.


This tutorial walks you through the fundamentals of Deep Reinforcement
Learning. At the end, you will implement an AI-powered Mario (using
`Double Deep Q-Networks <https://arxiv.org/pdf/1509.06461.pdf>`__) that
can play the game by itself.

Although no prior knowledge of RL is necessary for this tutorial, you
can familiarize yourself with these RL
`concepts <https://spinningup.openai.com/en/latest/spinningup/rl_intro.html>`__,
and have this handy
`cheatsheet <https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N>`__
as your companion. The full code is available
`here <https://github.com/yuansongFeng/MadMario/>`__.

.. figure:: /_static/img/mario.gif
   :alt: mario

"""


######################################################################
#
#

import torch
from torch import nn
from torchvision import transforms as T
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros


######################################################################
# RL Definitions
# """"""""""""""""""
#
# **Environment** The world that an agent interacts with and learns from.
#
# **Action** :math:`a` : How the Agent responds to the Environment. The
# set of all possible Actions is called *action-space*.
#
# **State** :math:`s` : The current characteristic of the Environment. The
# set of all possible States the Environment can be in is called
# *state-space*.
#
# **Reward** :math:`r` : Reward is the key feedback from Environment to
# Agent. It is what drives the Agent to learn and to change its future
# action. An aggregation of rewards over multiple time steps is called
# **Return**.
#
# **Optimal Action-Value function** :math:`Q^*(s,a)` : Gives the expected
# return if you start in state :math:`s`, take an arbitrary action
# :math:`a`, and then for each future time step take the action that
# maximizes returns. :math:`Q` can be said to stand for the “quality” of
# the action in a state. We try to approximate this function.
#


######################################################################
# Environment
# """"""""""""""""
#
# Initialize Environment
# ------------------------
#
# In Mario, the environment consists of tubes, mushrooms and other
# components.
#
# When Mario makes an action, the environment responds with the changed
# (next) state, reward and other info.
#

# Initialize Super Mario environment
from Mario_RL.Logging import MetricLogger
from Mario_RL.Mario import Mario
from Mario_RL.Wrapper import SkipFrame, GrayScaleObservation, ResizeObservation

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

######################################################################
# Let’s play!
# """""""""""""""
#
# In this example we run the training loop for 10 episodes, but for Mario to truly learn the ways of
# his world, we suggest running the loop for at least 40,000 episodes!
#
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 100
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


######################################################################
# Conclusion
# """""""""""""""
#
# In this tutorial, we saw how we can use PyTorch to train a game-playing AI. You can use the same methods
# to train an AI to play any of the games at the `OpenAI gym <https://gym.openai.com/>`__. Hope you enjoyed this tutorial, feel free to reach us at
# `our github <https://github.com/yuansongFeng/MadMario/>`__!
