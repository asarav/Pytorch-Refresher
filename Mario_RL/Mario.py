import copy
from collections import deque
import random

import numpy as np
import torch
from torch import nn

######################################################################
# In the following sections, we will populate Mario’s parameters and
# define his functions.
#


######################################################################
# Act
# --------------
#
# For any given state, an agent can choose to do the most optimal action
# (**exploit**) or a random action (**explore**).
#
# Mario randomly explores with a chance of ``self.exploration_rate``; when
# he chooses to exploit, he relies on ``MarioNet`` (implemented in
# ``Learn`` section) to provide the most optimal action.
#


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.
    Inputs:
    state(LazyFrame): A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx


######################################################################
# Cache and Recall
# ----------------------
#
# These two functions serve as Mario’s “memory” process.
#
# ``cache()``: Each time Mario performs an action, he stores the
# ``experience`` to his memory. His experience includes the current
# *state*, *action* performed, *reward* from the action, the *next state*,
# and whether the game is *done*.
#
# ``recall()``: Mario randomly samples a batch of experiences from his
# memory, and uses that to learn the game.
#


class Mario(Mario):  # subclassing for continuity
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


######################################################################
# Learn
# --------------
#
# Mario uses the `DDQN algorithm <https://arxiv.org/pdf/1509.06461>`__
# under the hood. DDQN uses two ConvNets - :math:`Q_{online}` and
# :math:`Q_{target}` - that independently approximate the optimal
# action-value function.
#
# In our implementation, we share feature generator ``features`` across
# :math:`Q_{online}` and :math:`Q_{target}`, but maintain separate FC
# classifiers for each. :math:`\theta_{target}` (the parameters of
# :math:`Q_{target}`) is frozen to prevent updation by backprop. Instead,
# it is periodically synced with :math:`\theta_{online}` (more on this
# later).
#
# Neural Network
# ~~~~~~~~~~~~~~~~~~


class MarioNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


######################################################################
# TD Estimate & TD Target
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Two values are involved in learning:
#
# **TD Estimate** - the predicted optimal :math:`Q^*` for a given state
# :math:`s`
#
# .. math::
#
#
#    {TD}_e = Q_{online}^*(s,a)
#
# **TD Target** - aggregation of current reward and the estimated
# :math:`Q^*` in the next state :math:`s'`
#
# .. math::
#
#
#    a' = argmax_{a} Q_{online}(s', a)
#
# .. math::
#
#
#    {TD}_t = r + \gamma Q_{target}^*(s',a')
#
# Because we don’t know what next action :math:`a'` will be, we use the
# action :math:`a'` maximizes :math:`Q_{online}` in the next state
# :math:`s'`.
#
# Notice we use the
# `@torch.no_grad() <https://pytorch.org/docs/stable/generated/torch.no_grad.html#no-grad>`__
# decorator on ``td_target()`` to disable gradient calculations here
# (because we don’t need to backpropagate on :math:`\theta_{target}`).
#


class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


######################################################################
# Updating the model
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As Mario samples inputs from his replay buffer, we compute :math:`TD_t`
# and :math:`TD_e` and backpropagate this loss down :math:`Q_{online}` to
# update its parameters :math:`\theta_{online}` (:math:`\alpha` is the
# learning rate ``lr`` passed to the ``optimizer``)
#
# .. math::
#
#
#    \theta_{online} \leftarrow \theta_{online} + \alpha \nabla(TD_e - TD_t)
#
# :math:`\theta_{target}` does not update through backpropagation.
# Instead, we periodically copy :math:`\theta_{online}` to
# :math:`\theta_{target}`
#
# .. math::
#
#
#    \theta_{target} \leftarrow \theta_{online}
#
#


class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


######################################################################
# Save checkpoint
# ~~~~~~~~~~~~~~~~~~
#


class Mario(Mario):
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


######################################################################
# Putting it all together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)