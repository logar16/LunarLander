from collections import deque

import numpy as np
import torch


class ExperienceBuffer:
    """Stores a lot of data to help with experience replay.  Also can be handy for saving stats"""

    def __init__(self, total_size: int, obs_size: int, seq_len: int, batch_size: int, device='cpu'):
        """
        :param total_size: Maximum number of memories to keep
        :param obs_size: size of each observation
        :param seq_len: Number of sequential memories to store together (to give context)
        :param batch_size: minibatch sample size (number of memories to return)
        :param device: PyTorch device to use for storing memory.
        """
        if not (total_size and obs_size and seq_len and batch_size):
            raise ValueError()

        self.size = total_size
        self.num_input = obs_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.length = 0
        self.device = device
        # The Arrays
        self.start_states = torch.zeros((total_size, seq_len, obs_size), dtype=torch.float16, device=device)
        self.actions = torch.zeros(total_size, dtype=torch.int8, device=device)
        self.rewards = torch.zeros(total_size, dtype=torch.float16, device=device)
        self.next_states = torch.zeros_like(self.start_states)
        self.terminations = torch.zeros(total_size, dtype=torch.bool, device=device)
        self.episodic_memory = EpisodicMemory()

    def add(self, start, action, reward, next_state, done):
        """Add an experience for future use"""
        self.episodic_memory.add(reward)

        # Ignore some of the transitory states as we've got them in the buffer and they don't help that much
        if not done and np.random.rand() < 0.5:
            return

        i = self.length % self.size  # Overwrite older entries once you run out of room (circular buffer)
        self.start_states[i] = start
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminations[i] = done
        self.length += 1
        if done:
            self.episodic_memory.reset()

    def ready(self):
        return self.length > 2048

    def sample(self):
        """Create a randomized mini-batch using previous experience"""
        size = self.batch_size
        if self.length < size:
            raise ValueError("Buffer is not large enough to return a batch size of " + str(size))

        max_index = min(self.length, self.size)
        indexes = torch.randint(high=max_index, size=(size,), device=self.device)
        starts = self.start_states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        nexts = self.next_states[indexes]
        terminations = self.terminations[indexes]
        return starts, actions, rewards, nexts, terminations

    def get_statistics(self):
        return self.episodic_memory.statistics(cumulative=True)


class EpisodicMemory:
    """Keeps track of rewards (and potentially other stats) over an episode (as well as keeping some overall stats)."""

    def __init__(self):
        self.step = 0
        self.num_episodes = 0
        self.all_steps = deque()
        self.rewards = deque()
        self.all_rewards = deque()

    def add(self, reward):
        """Add the reward received for the last step"""
        self.step += 1
        self.rewards.append(reward)

    def reset(self):
        """Call this when an episode is over to save long term stats and clear the single-episode stats"""
        self.all_steps.append(self.step)
        self.all_rewards.append(sum(self.rewards))
        self.num_episodes += 1
        self.step = 0
        self.rewards.clear()

    def statistics(self, cumulative=False):
        """Return a nice array of the rewards experienced thus far"""
        if cumulative:
            return np.asarray(self.all_rewards), np.asarray(self.all_steps)
        return np.asarray(self.rewards), self.step

    def last_reward(self):
        return self.rewards[-1]
