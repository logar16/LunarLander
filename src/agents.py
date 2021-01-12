from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import gym
from typing import Type, Callable

from memory import ExperienceBuffer
import numpy as np
from builders import ModelBuilder
from copy import deepcopy


class DQNAgent:

    def __init__(self, gamma=0.9992,
                 rar=1.1, rar_decay=0.999997,
                 memory_size=120000, minibatch_size=100,
                 replay_freq=40, target_update=100000,
                 config: dict = None, num_inputs=8, num_actions=4,
                 optim_type: Type[optim.Optimizer] = optim.Adam, optim_args={'lr': 0.015},
                 criterion=nn.SmoothL1Loss(), device: str = 'cpu', seq_len: int = 3):
        """
        :param gamma: Discount factor for Bellman update
        :param rar: Random action rate (RAR)
        :param rar_decay: RAR Decay constant.  rar *= rar_decay every step until it reaches a sufficiently small number
        :param memory_size: How many experiences to keep for experience replay
        :param minibatch_size: Number of randomly selected experiences to use in a batch training
        :param replay_freq: How often experience replay should be run (i.e. how many episodes between replay)
        :param device: Defaults to "cpu" since this works fastest on my machine for single-runs,
        but if you set it to None it will try to infer whether to use "cuda" or "cpu".  You can also set to "cuda".
        """
        self.config = config
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        # Discount of future reward
        self.gamma = gamma
        # Random action setup
        self.random_action_rate = rar
        self.rar_decay = rar_decay
        # Memory for experience replay
        self.memory = ExperienceBuffer(memory_size, self.num_inputs, seq_len, minibatch_size, device=self.device)
        self.replay_freq = replay_freq
        # How many iterations between copying the active model to the target model
        self.update_target_freq = target_update

        self.active_model: nn.Module = None
        self.target_model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.optim_type: Type[optim.Optimizer] = optim_type
        self.optim_args: dict = optim_args
        self.criterion: nn.Module = criterion

        self.end_early = False
        self.total_episodes = 1000
        self.losses = []

    def set_model(self, model: nn.Module):
        """
        Give the agent a model to use.
        Will set the model to be active model, and a copy of it to be the target model.
        """
        if self.device == 'cuda':
            model = model.cuda()
        self.active_model = model
        self.optimizer = self.optim_type(model.parameters(), **self.optim_args)
        # copy active model to begin
        self.target_model = deepcopy(model)
        # TODO: DDQN?

    def create_model(self):
        model = ModelBuilder.build(num_inputs=self.num_inputs, num_actions=self.num_actions)
        self.set_model(model)

    def train(self, env: gym.Env, episodes: int, callback: Callable = None, verbose: bool = False):
        """
        Use for training the model.  See `self.run()` for evaluation
        """
        if verbose:
            print('Device:', self.device)

        if not self.active_model:
            self.create_model()

        self.total_episodes = episodes
        percent = (episodes / 100) or 1
        step = 0

        for i in range(episodes):
            if self.end_early:
                break
            # Setup environment and start state
            state = env.reset()
            state = torch.tensor(state, device=self.device)
            # Memory buffer
            buffer = torch.zeros(self.memory.seq_len, self.num_inputs, device=self.device)
            buffer[-1] = state

            done = False
            while not done:
                step += 1
                # Pick an action
                action = self.query(buffer)
                # See where the action takes you
                state, reward, done, info = env.step(action)
                state = torch.tensor(state, device=self.device)
                # Add next state to the buffer (and remove oldest data)
                new_buffer = torch.cat((buffer[1:], state.unsqueeze(0)))
                # Add the transition to the memory.
                self.memory.add(buffer, action, reward, new_buffer, done)
                buffer = new_buffer

                # Check if it is time to learn
                if step % self.replay_freq == 0 and self.memory.ready():
                    self.experience_replay()

                # Update the target to reflect knowledge
                if step % self.update_target_freq == 0:
                    self.target_model = deepcopy(self.active_model)
                    # if verbose: print(f'\nCopying Active to Target on step {step}')

            if self.memory.ready():
                self.experience_replay()

            if (i + 1) % percent == 0:
                self.update_event(int(i / percent), callback, verbose)

        self.update_event(100, callback, verbose)
        return self.memory.get_statistics()

    def update_event(self, percent: int, callback: Callable = None, verbose: bool = False):
        data = {
            'percent': percent,
            'stats': self.memory.get_statistics(),
            'rar': self.random_action_rate,
            'losses': torch.tensor(self.losses),
            'verbose': verbose,
        }
        if callback:
            callback(self, data)
        elif verbose:
            print(data)

    def evaluate(self, env: gym.Env, episodes: int, render=False, verbose=False, interactive=False):
        """
        Use when testing a pre-trained model. Will not update/train the model.
        """
        if verbose:
            print('Device:', self.device)
        rewards = deque()
        for i in range(episodes):
            # Setup environment and start state
            state = env.reset()
            state = torch.tensor(state, device=self.device)
            # Memory buffer
            buffer = torch.zeros(self.memory.seq_len, self.num_inputs, device=self.device)
            buffer[-1] = state

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
                action = self.query(buffer, random_actions=False)
                # See where the action takes you
                state, reward, done, info = env.step(action)
                state = torch.tensor(state, device=self.device)
                buffer = torch.cat((buffer[1:], state.unsqueeze(0)))
                total += reward
                # if verbose:
                #     print(f"Action: {action}, Reward: {round(reward, 3)}, Total: {round(total, 4)}")

            rewards.append(total)
            if verbose:
                print(f"{i} Final Reward: {total}")
        rewards = np.asarray(rewards)
        print("Average Reward:", rewards.mean())
        print("Standard Dev:", rewards.std())
        return rewards

    def query(self, buffer, random_actions=True):
        """Ask the agent what action to take given the state"""
        if random_actions:
            # Decrement random action rate slightly, but only until it is pretty small, then leave it alone
            if self.random_action_rate > 0.01:
                self.random_action_rate *= self.rar_decay
            if np.random.random() < self.random_action_rate:
                return np.random.randint(self.num_actions)  # Pick a random action

        with torch.no_grad():
            results = self.active_model(buffer.unsqueeze(0))
            actions = results.squeeze()
            return torch.argmax(actions).item()

    def experience_replay(self):
        """
        Train the model using experience replay (take minibatches from the replay memory)
        """
        prev_states, actions, rewards, next_states, terminates = self.memory.sample()
        size = actions.size(0)

        # Convert to Torch Tensors
        prev_states = prev_states.to(dtype=torch.float)
        next_states = next_states.to(dtype=torch.float)

        # TODO: DDQN
        # if np.random.randint(2):
        #     update_model = self.active_model
        #     ref_model = self.active_model2
        # else:
        #     update_model = self.active_model2
        #     ref_model = self.active_model

        # Use self.target_model for the expected future value (since it is more stable)
        prev_state_value = self.active_model(prev_states)
        next_state_value = self.target_model(next_states)
        # Update terminating transitions to be equal to the reward of ending
        # Update non-termination transitions to be equal to:
        #   the reward + discounted expected return for next best action (Bellman update)
        # Termination is the distinguishing factor
        non_terminal_values = ~terminates * self.gamma * next_state_value.max(axis=1).values
        updated_value = prev_state_value.clone()
        action_indices = actions.to(dtype=torch.long)
        updated_value[torch.arange(size), action_indices] = rewards + non_terminal_values
        # Update the model (leave target alone for now)
        self.train_model(prev_states, updated_value)

    def train_model(self, X, y):
        """
        Give the model some experiences and let it train/learn how to predict outcomes from states.
        :param X: Input states
        :param y: Actual rewards
        """
        self.optimizer.zero_grad()
        # Have it guess what the reward would be for a state
        output = self.active_model(X)
        # Compare to actual state
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

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
        filename = f'models/networks/{save_name}_{suffix}.pt'
        print('Saving', filename)
        torch.save(self.active_model.state_dict(), filename)

    def load(self, filename):
        """Load the model and set it as active (and target) model"""
        print('Loading', filename)
        if not self.active_model:
            self.create_model()
        state = torch.load(filename)
        self.active_model.load_state_dict(state)
        self.set_model(self.active_model)
