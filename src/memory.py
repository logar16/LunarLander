from collections import deque

import numpy as np


class ExperienceBuffer:
    """Stores a lot of data to help with experience replay.  Also can be handy for saving stats"""

    def __init__(self, total_size, num_input, batch_size):
        if not total_size or not num_input or not batch_size:
            raise ValueError()

        self.size = total_size
        self.batch_size = batch_size
        self.num_input = num_input
        self.length = 0
        # The Arrays
        self.start_states = np.ndarray((total_size, num_input), dtype=np.float16)
        self.actions = np.ndarray(total_size, dtype=np.int8)
        self.rewards = np.ndarray(total_size, dtype=np.float16)
        self.next_states = np.ndarray((total_size, num_input), dtype=np.float16)
        self.terminations = np.ndarray(total_size, dtype=np.bool)
        self.episodic_memory = EpisodicMemory()

    def add(self, start, action, reward, next_state, done):
        """Add an experience for future use"""
        i = self.length % self.size  # Overwrite older entries once you run out of room
        self.start_states[i] = start
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminations[i] = done
        self.length += 1
        self.episodic_memory.add(reward)
        if done:
            self.episodic_memory.reset()

    def ready(self):
        return self.length > 20000

    def sample(self, replacement=False):
        """Create a randomized mini-batch using previous experience"""
        size = self.batch_size
        if self.length < size:
            raise ValueError("Buffer is not large enough to return a batch size of " + str(size))

        max_index = min(self.length, self.size)
        indexes = np.random.choice(max_index, size, replace=replacement)
        starts = self.start_states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        nexts = self.next_states[indexes]
        terminations = self.terminations[indexes]
        return starts, actions, rewards, nexts, terminations

    def get_statistics(self):
        return self.episodic_memory.statistics(cumulative=True)

    def fake_experiences(self, iterations):
        state = np.arange(self.num_input)
        for i in range(iterations):
            next_state = np.random.choice(10, self.num_input)
            action = np.random.randint(4)
            reward = np.random.random()
            done = np.random.random() < 0.05
            self.add(state, action, reward, next_state, done)


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
