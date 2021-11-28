import gym
import numpy as np
import torch
from torchvision import transforms as T
from gym.spaces import Box

######################################################################
# Preprocess Environment
# ------------------------
#
# Environment data is returned to the agent in ``next_state``. As you saw
# above, each state is represented by a ``[3, 240, 256]`` size array.
# Often that is more information than our agent needs; for instance,
# Mario_RL’s actions do not depend on the color of the pipes or the sky!
#
# We use **Wrappers** to preprocess environment data before sending it to
# the agent.
#
# ``GrayScaleObservation`` is a common wrapper to transform an RGB image
# to grayscale; doing so reduces the size of the state representation
# without losing useful information. Now the size of each state:
# ``[1, 240, 256]``
#
# ``ResizeObservation`` downsamples each observation into a square image.
# New size: ``[1, 84, 84]``
#
# ``SkipFrame`` is a custom wrapper that inherits from ``gym.Wrapper`` and
# implements the ``step()`` function. Because consecutive frames don’t
# vary much, we can skip n-intermediate frames without losing much
# information. The n-th frame aggregates rewards accumulated over each
# skipped frame.
#
# ``FrameStack`` is a wrapper that allows us to squash consecutive frames
# of the environment into a single observation point to feed to our
# learning model. This way, we can identify if Mario_RL was landing or
# jumping based on the direction of his movement in the previous several
# frames.
#

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation