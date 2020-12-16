import gc
import time

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from builders import RandomAgentBuilder, RandomModelBuilder, SpecificAgentBuilder
from agents import DQNAgent


class PerformanceTracker:
    verbose = False

    def __init__(self, agent, model, id_, verbose=False):
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
        self.id = id_
        self.agent = agent
        self.agent_params = agent.config
        if agent.active_model:
            model = agent.active_model
        else:
            agent.set_model(model)

        self.model = model
        self.architecture = str(model)
        self.layers = [v for k, v in model.named_children()]

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
        if self.verbose:
            print("Parameters:", self.agent_params)
            print("Layers:", self.model)

        print(f'Starting at: {time.strftime("%H:%M:%S")}')
        start = time.time()
        self.rewards, self.steps = self.agent.train(env,
                                                    episodes,
                                                    callback=self.on_progress,
                                                    verbose=self.verbose)
        print(f"\nCompleted in {round(time.time() - start, 4)} sec. "
              f"at {time.strftime('%H:%M:%S')}")
        self.end()
        print(self)


    def on_progress(self, agent: DQNAgent, data):
        """
        After 1% of the total iterations is complete, the agent will call this function
        This is an opportunity to decide if it is time to quit early.
        """
        percent: int = data['percent']
        reward, steps = data['stats']
        rar = data['rar']

        if len(reward) >= 100:
            last100 = reward[-100:]
            mean = np.round(last100.mean())
            if mean >= 200:
                print("Successfully completed goal")
                self.success = True
                self.exit_early = True
                agent.end_training_early()
            elif mean >= 50 and percent % 5 == 0:
                print("\nGood performance found, saving checkpoint")
                epoch = int(self.episodes * percent / 100)
                agent.save(f'{self.id}', f'{epoch}_{mean}')

        if self.verbose and percent % 10 == 0:
            # TODO: Print additional info
            print(f"\n{percent}% "
                  f"\tTotal reward={round(reward.mean(), 3)}  "
                  f"steps={steps.sum()}  "
                  f"rar={round(rar, 3)}")
            # look at the last several episodes
            reward = reward[-self.percent_size:]
            print(f"\t\tRecent reward={round(reward.mean(), 3)},  "
                  f"max={round(reward.max(), 3)}")

        if self.verbose:
            print(f'{percent}% ... ', end="")
        else:
            progress = '=' * int(percent)
            progress += '>'
            left = ' ' * (100 - percent)
            print(f'{percent}% [{progress + left}]', end='\r')


    def end(self):
        """Save and clean up"""
        self.epochs = self.rewards.reshape(-1, 100)
        epoch_counts = np.arange(self.epochs.shape[0])
        epoch_avgs = self.epochs.mean(axis=1)
        self.epoch_avgs = np.round(epoch_avgs)
        self.positive_epochs = epoch_counts[epoch_avgs > 0]
        self.great_epochs = epoch_counts[epoch_avgs > 100]
        self.best_epoch = epoch_avgs.argmax()
        self.best_epoch_avg = np.round(epoch_avgs[self.best_epoch])

        last100 = self.rewards[-100:]
        self.last100 = last100
        self.avg = np.round(last100.mean(), 1)
        self.std = np.round(last100.std(), 1)
        if self.avg > 0:
            self.agent.save('nn', str(self.avg))
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
        string += f"\nLast Epoch Mean: {self.avg}"
        if len(self.positive_epochs):
            string += f"\n# Positive Epochs: {len(self.positive_epochs)}; " \
                      f"Epoch Indices: {self.positive_epochs}"
            string += f"\n# Great Performing (>100 avg) Epochs: {len(self.great_epochs)}; " \
                      f"Epoch Indices: {self.great_epochs}"
        string += f"\nBest Epoch was #{self.best_epoch} (of {self.epochs.shape[0]}) " \
                  f"with avg reward: {self.best_epoch_avg}"

        if self.success:
            string += "\n!!!Success!!!"
        if self.exit_early:
            string += "\n(Exited Early for poor performance)"
        return string


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
        total = 20
        print(f"Iterating over {total} random samples...")
        for i in range(1, total + 1):
            print(f"\n\n**Training Agent {i}/{total}**")
            agent = self.agent_builder.next()
            model = self.model_builder.next()
            tracked_agent = PerformanceTracker(agent, model, i, self.verbose)
            self.run_agent(tracked_agent)
            print(f"**Finished Agent {i}/{total}**\n")
        print(f"Done iterating over {total} random samples")


    def run_agent(self, tracked_agent: PerformanceTracker):
        tracked_agent.train(self.env, self.episodes)
        self.save_report(tracked_agent)
        del tracked_agent   # Clean up the memory
        gc.collect()

    def save_report(self, tracked_agent: PerformanceTracker):
        """
        Document the performance of the agent in graphs and by other means
        """
        # Table #
        self.save_table(tracked_agent)

        # Figures #
        self.save_figures(tracked_agent)

    def save_table(self, tracker: PerformanceTracker):
        """
        Append details to the results.csv file for easier tracking
        """
        params: dict = tracker.agent_params
        # Best Epoch, Best Epoch Avg
        details = f'{tracker.best_epoch},{tracker.best_epoch_avg},'
        # Positive Epochs
        details += f'{len(tracker.positive_epochs)},'
        # Final Epoch Avg, Final Epoch StdDev
        details += f'{tracker.avg},{tracker.std},'
        # Layer 1, Layer 2 architectures
        layers = tracker.layers
        details += f'{layers[0].out_features},{layers[2].out_features},'
        # Gamma, Random Action Rate, RAR Decay
        details += f'{params["gamma"]},{params["rar"]},{params["rar_decay"]},'
        # Memory Size, Minibatch Size
        details += f'{params["memory_size"]},{params["minibatch_size"]},'
        # Replay Freq, Target Update,
        details += f'{params["replay_freq"]},{params["target_update"]},'
        # Optimizer
        optimizer = str(params["optim_type"])
        start = optimizer.rfind('.') + 1
        end = optimizer.rfind("'")
        optimizer = optimizer[start:end]
        details += f'{optimizer},'
        # Learning Rate, Other Optimizer Args
        lr = params["optim_args"]["lr"]
        # del params["optim_args"]["lr"]
        # others = params["optim_args"] if len(params["optim_args"]) else "N/A"
        details += f'{lr}'  # ,{others}
        # Append to file
        with open('models/results.csv', 'a') as file:
            file.write(f'{details}\n')

    def save_figures(self, tracker: PerformanceTracker):
        """
        Save figures showing progress of agent
        """
        plt.clf()  # Clear the way for the next one
        rewards = np.asarray(tracker.rewards)
        rewards[rewards < -500] = -500  # Throws off the graph when the occasional point goes really negative

        plt.title('Reward per Training Episode')
        plt.plot(rewards, 'bo', label='Episodic Reward')
        plt.plot(self.moving_average(rewards), 'r', label='Rolling Mean (20)')
        plt.axhline(200, 0, 1, color='g')
        plt.ylabel('Reward')
        plt.xlabel('Episode')

        name = f'{tracker.id}_{round(rewards[-100:].mean(), 2)}'
        plt.savefig(f'figures/{name}_reward.png')

        plt.clf()
        cumulative = np.cumsum(rewards)
        plt.title('Cumulative Reward vs. Target Update')
        plt.plot(cumulative, label='Cumulative Reward')

        update_freq = tracker.agent_params['target_update']
        steps = np.cumsum(tracker.steps)
        for i in range(steps.size):
            step = steps[i]
            if step > update_freq:
                steps[i:] -= update_freq
                plt.axvline(i, 0, 1, color='r')
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.savefig(f'figures/{name}_cumulative.png')

    def moving_average(self, array, n=20):
        ret = np.cumsum(array, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def resume(self):
        """Load in a file and start where you left off"""
        pass
