from collections import deque

import torch
import torch.nn as nn

import gym
import typing

from memory import ExperienceBuffer
import numpy as np
from builders import ModelBuilder
from copy import deepcopy


class DQNAgent:

    def __init__(self, gamma=0.9992,
                 rar=1.1, rar_decay=0.999997,
                 memory_size=120000, minibatch_size=100,
                 replay_freq=40, target_update=100000,
                 config=None, num_inputs=8, num_actions=4):
        """
        :param gamma: Discount factor for Bellman update
        :param rar: Random action rate (RAR)
        :param rar_decay: RAR Decay constant.  rar *= rar_decay every step until it reaches a sufficiently small number
        :param memory_size: How many experiences to keep for experience replay
        :param minibatch_size: Number of randomly selected experiences to use in a batch training
        :param replay_freq: How often experience replay should be run (i.e. how many episodes between replay)
        """
        self.config = config
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        # Discount of future reward
        self.gamma = gamma
        # Random action setup
        self.random_action_rate = rar
        self.rar_decay = rar_decay
        # Memory for experience replay
        self.memory = ExperienceBuffer(memory_size, self.num_inputs, minibatch_size)
        self.replay_freq = replay_freq
        # How many iterations between copying the active model to the target model
        self.update_target_freq = target_update

        self.active_model = self.target_model = None
        self.end_early = False
        self.total_episodes = 1000

    def set_model(self, model: nn.Module):
        """
        Give the agent a model to use.
        Will set the model to be active model, and a copy of it to be the target model.
        """
        self.active_model = model
        # copy active model to begin
        self.target_model = deepcopy(model)
        # TODO: DDQN?

    def create_model(self):
        model = ModelBuilder.build(num_inputs=self.num_inputs, num_actions=self.num_actions)
        self.set_model(model)

    def train(self, env: gym.Env, episodes: int, callback: typing.Callable = None):
        if not self.active_model:
            self.create_model()
        self.total_episodes = episodes
        percent = (episodes / 100) or 1
        step = 0

        for i in range(episodes):
            if self.end_early:
                break
            # setup environment and start state
            state = env.reset()
            state = np.asarray(state)
            done = False

            while not done:
                step += 1
                # Pick an action
                action = self.query(state)
                # See where the action takes you
                next_state, reward, done, info = env.step(action)
                next_state = np.asarray(next_state)
                # Add the transition to the memory.
                self.memory.add(state, action, reward, next_state, done)
                state = next_state

                # Check if it is time to learn
                if step % self.replay_freq == 0 and self.memory.ready():
                    self.experience_replay()

                # Update the target to reflect knowledge
                if step % self.update_target_freq == 0:
                    self.target_model = deepcopy(self.active_model)
                    # TODO: Verbose print

            if self.memory.ready():
                self.experience_replay()

            if i % percent == 0:
                if callback:
                    data = {'percent': (i / percent), 'stats': self.memory.get_statistics(),
                            'rar': self.random_action_rate}
                    callback(self, data)

        return self.memory.get_statistics()

    def evaluate(self, env: gym.Env, episodes: int, render=False, verbose=False, interactive=False):
        """
        Use when testing a pre-trained model. Will not update/train the model.
        """
        rewards = deque()
        for i in range(episodes):
            # setup environment and start state
            state = env.reset()
            state = np.asarray(state)
            done = False
            total = 0
            skip = 0
            while not done:
                if render:
                    env.render()
                    if interactive:
                        print("State:", state.round(3))
                        if skip > 0:
                            skip -= 1
                        else:
                            entry = input("(Enter number of frames to skip) ")
                            if entry.isalnum():
                                skip = int(entry)

                # Pick an action
                action = self.query(state, random_actions=False)
                # See where the action takes you
                next_state, reward, done, info = env.step(action)
                state = np.asarray(next_state)
                total += reward
                if verbose:
                    print("Action:", action, "  Reward:", round(reward, 3), "  Total:", round(total, 4))

            rewards.append(total)
            print(i, " Final Reward:", total)
        rewards = np.asarray(rewards)
        print("Average Reward:", rewards.mean())
        print("Standard Dev:", rewards.std())
        return rewards

    def query(self, state, random_actions=True):
        """Ask the agent what action to take given the state"""
        if random_actions:
            # Decrement random action rate slightly, but only until it is pretty small, then leave it alone
            if self.random_action_rate > 0.001:
                self.random_action_rate *= self.rar_decay
            if np.random.random() < self.random_action_rate:
                return np.random.randint(self.num_actions)  # Pick a random action
        state = np.asarray(state).reshape(1, self.num_inputs)
        actions = self.active_model.predict(state)
        return np.argmax(actions)

    def experience_replay(self):
        """Train the model using experience replay (take minibatches from the replay memory)"""
        prev_states, actions, rewards, next_states, terminates = self.memory.sample()
        size = actions.size

        # DDQN
        # if np.random.randint(2):
        #     update_model = self.active_model
        #     ref_model = self.active_model2
        # else:
        #     update_model = self.active_model2
        #     ref_model = self.active_model

        # Use self.target_model for the expected future value (since it is more stable)
        prev_state_value = self.active_model.predict(prev_states)
        next_state_value = self.target_model.predict(next_states)
        # Update terminating transitions to be equal to the reward of ending
        # Update non-termination transitions to be equal to:
        #   the reward + discounted expected return for next best action (Bellman update)
        # Termination is the distinguishing faction
        non_terminal_values = ~terminates * self.gamma * next_state_value.max(axis=1)
        updated_value = prev_state_value.copy()
        updated_value[np.arange(size), actions] = rewards + non_terminal_values
        # Update the model (leave target alone for now)
        self.active_model.train_on_batch(prev_states, updated_value)

    def end_training_early(self):
        self.end_early = True

    def __repr__(self):
        string = ""
        if self.config:
            string = f"Config: {self.config}\n"
        # string += f"Model:\n {self.active_model}"
        return string

    def save(self, save_name='', suffix=''):
        """
        Save the model for later loading
        """
        filename = f'models/{save_name}_{suffix}.pt'
        print('Saving', filename)
        torch.save(self.active_model.state_dict, filename)

    def load(self, filename):
        """Load the model and set it as active (and target) model"""
        print('Loading', filename)
        self.create_model()
        self.active_model.load_state_dict(torch.load(filename))
        self.set_model(self.active_model)
