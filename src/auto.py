import gc
import time

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from builders import RandomAgentBuilder, RandomModelBuilder, SpecificAgentBuilder
from agents import DQNAgent


class Runner:

    def __init__(self, env, all_perm, episodes=1000, verbose=False):
        self.all_perms = all_perm
        self.env = env
        self.episodes = episodes
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n
        # self.agent_builder = SpecificAgentBuilder(num_inputs, num_actions)
        self.agent_builder = RandomAgentBuilder(num_inputs, num_actions)
        self.model_builder = RandomModelBuilder(num_inputs, num_actions)
        self.verbose = verbose


    def start(self):
        if self.all_perms:
            self.iterate_all()
        else:
            self.iterate_randomly()


    def iterate_all(self):
        print("Iterating over all conceivable combinations...")
        print("Done iterating over all conceivable combinations")
        # for model in self.model_builder:
        #     for agent in self.agent_builder:
        #         agent.model = model
        #         self.run_agent(agent)

    def iterate_randomly(self):
        total = 50
        # percent = total / 100
        print("Iterating over {} random samples...".format(total))
        for i in range(total):
            # self.print_complete(i * percent)
            agent = self.agent_builder.next()
            model = self.model_builder.next()
            agent = PerformanceTracker(agent, model, self.verbose)
            self.run_agent(agent)
        print("Done iterating over {} random samples".format(total))


    def run_agent(self, agent):
        agent.train(self.env, self.episodes)
        self.save_report(agent)
        del agent   # Clean up the memory
        gc.collect()

    def save_report(self, agent):
        """Document the performance of an agent"""
        plt.clf()   # Clear the way for the next one
        rewards = np.asarray(agent.rewards)
        # plt.plot(rewards, 'g', label='?')
        plt.title('Reward per Training Episode')
        plt.plot(rewards, 'bo', label='Episodic Reward')
        plt.plot(self.moving_average(rewards), 'r', label='Rolling Mean (20)')
        plt.axhline(200, 0, 1, color='g')
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        name = round(rewards[-100:].mean(), 2)
        name = str(name) + '_fig.png'
        plt.savefig('figures/reward_' + name)

        plt.clf()
        cumulative = np.cumsum(rewards)
        plt.title('Cumulative Reward vs. Target Update')
        plt.plot(cumulative, label='Cumulative Reward')

        update_freq = agent.agent_params['target_update']
        steps = np.cumsum(agent.steps)
        for i in range(steps.size):
            step = steps[i]
            if step > update_freq:
                steps[i:] -= update_freq
                plt.axvline(i, 0, 1, color='r')

        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.savefig('figures/cumulative_' + name)

    def moving_average(self, array, n=20):
        ret = np.cumsum(array, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def resume(self):
        """Load in a file and start where you left off"""
        pass


class PerformanceTracker:
    verbose = False

    def __init__(self, agent, model, verbose=False):
        """
        Mostly used for convenience in tracking the stats of an agent

        :param agent: The agent to track
        :type agent DQNAgent
        :param model: The model the agent uses
        :type model Sequential
        """
        if not model or not agent:
            raise TypeError()

        self.success = False
        self.agent = agent
        self.agent_params = agent.config
        if agent.active_model:
            model = agent.active_model
        else:
            agent.set_model(model)

        self.model = model
        self.architecture = str(model)

        self.verbose = verbose
        # self.mean_max = np.ndarray((100, 2))  # Covers 0-99%  mean() and max()
        self.episodes = 1000  # Temp value
        self.exit_early = False
        self.rewards = None
        self.steps = None
        self.percent_size = 10


    def train(self, env, episodes):
        """Tell the agent to start training.  Will exit early if performance declines too quickly"""
        self.episodes = episodes
        self.percent_size = episodes // 100
        print("\n\n**Training Agent**")
        if self.verbose:
            print("Parameters:", self.agent_params)
            print("Layers:", self.model)

        print(time.strftime('%H:%M:%S'))
        start = time.time()
        self.rewards, self.steps = self.agent.train(env, episodes, callback=self.on_progress)
        print(f"Completed in {time.time() - start} sec. at {time.strftime('%H:%M:%S')}")
        self.end()
        print(self, "\n\n")


    def on_progress(self, agent, data):
        """After 1% of the total iterations is complete, the agent will call this function
        This is an opportunity to decide if it is time to quit early.
        """
        percent = data['percent']
        reward, steps = data['stats']
        rar = data['rar']

        if len(reward) >= 100:
            last100 = reward[-100:]
            if last100.mean() >= 200:
                print("Successfully completed goal")
                self.success = True
                self.exit_early = True
                agent.end_training_early()

        if self.verbose and percent % 10 == 0:
            print(f"\n{percent}% "
                  f"\tTotal reward={round(reward.mean(), 3)}  steps={steps.sum()}  rar={round(rar, 3)}")
            # look at the last few episodes
            reward = reward[-self.percent_size:]
            print(f"\t\tRecent reward={round(reward.mean(), 3)},  max={round(reward.max(), 3)}")
        print(f'{percent}% ... ', end="")


    def end(self):
        """Save and clean up"""
        last100 = self.rewards[-100:]
        self.last100 = last100
        self.avg = last100.mean()
        self.std = last100.std()
        if self.avg > 0:
            rounded = round(self.avg, 2)
            self.agent.save('nn', str(rounded))
            if self.avg >= 200:
                self.success = True

        # clear up some space in memory
        del self.agent
        del self.model

    def __repr__(self):
        string = f"Model: {self.architecture}"
        string += f"\nParameters: {self.agent_params}"
        string += f"\nEpisodes: {self.episodes}"
        string += f"\nStandard Dev: {self.std}"
        string += f"\nLast 100 Mean: {self.avg}"
        if self.success:
            string += "\n!!!Success!!!"
        if self.exit_early:
            string += "\n(Exited Early)"
        return string
