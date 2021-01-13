import sys
import argparse
import time
import gym
import numpy as np
# import matplotlib as mpl
import torch

# mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('src/')
from agents import DQNAgent
from auto import Runner
from builders import AgentLoader

env = gym.make('LunarLander-v2')
env._max_episode_steps = 600    # Makes the infinite flight episodes shorter (10 seconds at 60 FPS)
# env = gym.make('CartPole-v0')
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

run_id = time.strftime("%H-%M-%S")


def setup(config: str, load_file: str) -> DQNAgent:
    if config:
        loader = AgentLoader(config, num_actions=num_actions, num_inputs=num_inputs)
        agent = loader.load()
    else:
        agent = DQNAgent(num_actions=num_actions, num_inputs=num_inputs)
    if load_file:
        print(f'Loading "{load_file}"...')
        agent.load(load_file)
    return agent


def explore_options(iterations, verbose):
    runner = Runner(env, False, iterations, verbose)
    runner.start()


def print_progress(agent: DQNAgent, data: dict):
    percent = data['percent']
    progress = '=' * int(percent)
    progress += '>'
    left = ' ' * (100 - percent)
    progress = f'{percent}% [{progress + left}]'

    reward, steps = data['stats']
    mean = round(reward.mean(), 1)
    std = round(reward.std(), 1)
    positive = reward[reward > 0].size
    total = reward.size
    steps = steps.sum()
    losses = data['losses']

    if total > 50:
        graph(reward, verbose=True)
        plt.savefig(f'figures/{run_id}_training.png')
        if len(losses) > 10:
            graph(losses.detach().numpy(), xlabel='Replays', ylabel='Loss', window=5)
            plt.savefig(f'figures/{run_id}_losses.png')
    # print(progress + f'  μ: {mean}, σ: {std}; +{positive}/{total}, steps: {steps}', end='\r')
    # if percent % 5 != 0:
    #     return
    last100 = reward[-100:]
    last_mean = round(last100.mean(), 2)
    last_std = round(last100.std(), 1)
    verbose = data['verbose']

    if percent % 2 == 0 and last_mean > 200:
        print(' ' * 100, end='\r')
        if verbose:
            print('Last 100 episodes average over 200! ', end='')
        agent.save(f'{run_id}_{percent}p', str(round(last_mean, 0)))

    # rar = f'rar: {round(data["rar"], 5)}' if verbose else ''
    # Spaces at the end are to clean up the progress bar
    print(f'Total mean: {mean}, std: {std};  '
          f'Last 100 mean: {last_mean}, std: {last_std};  '
          f'Positive: {positive}/{total}  '
          f'Steps: {steps}  ',
          # rar,
          " " * 20)
    if verbose:
        if len(losses) > 1:
            mean = round(losses.mean().item(), 3)
            std = round(torch.std(losses).item(), 3)
            print(f'Recent Losses: {losses[-5:]}, mean: {mean}, std: {std}')
    print(progress, end='\r')


def moving_average(array: np.ndarray, n: int = 20):
    avg = np.cumsum(array, dtype=float)
    avg[n:] = avg[n:] - avg[:-n]
    avg = avg[(n - 1):] / n
    avg = np.append(array[:n - 1], avg)
    return avg


def moving_std(array: np.ndarray, n: int = 20):
    stride = array.strides[0]
    std = np.lib.stride_tricks.as_strided(array, shape=(array.size - n, n), strides=(stride, stride))
    std = np.std(std, axis=1)
    repeat = np.full(n, std[-1])
    std = np.append(std, repeat)
    return std


def save_rewards(filename, rewards, steps=None, window=20, verbose=False):
    # Rewards by episode
    plt.clf()
    rewards[rewards < -500] = -500
    graph(rewards, window=window, verbose=verbose)
    plt.savefig(f'figures/{filename}_episodes.png')
    # plt.show()

    # Rewards by total steps taken
    if steps is None:
        return
    steps = np.cumsum(steps)
    graph(rewards, steps, 'Steps', window=window, verbose=verbose)
    plt.savefig(f'figures/{filename}_steps.png')
    # plt.show()


def graph(rewards, steps=None, xlabel='Episode', ylabel='Reward', window: int = 20, verbose=False):
    if steps is None:
        steps = np.arange(len(rewards))
    average = moving_average(rewards, window)
    error = moving_std(rewards, window)
    plt.clf()
    positives = rewards > 0
    if verbose:
        plt.plot(steps[positives], rewards[positives], 'o', label='Positive Episode', color='tab:purple')
    steps = steps[window:]
    average = average[window:]
    error = error[window:]
    plt.plot(steps, average, label=f'Rolling Mean ({window})', color='tab:blue')
    plt.fill_between(steps, average - error, average + error, alpha=0.4, color='tab:blue')
    if ylabel == 'Reward':
        plt.axhline(200, 0, 1, color='limegreen')
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


def trim(s: str, start: str, end: str) -> str:
    return s[(s.rfind(start) + 1):s.rfind(end)]


def main(arguments):
    interactive = arguments.interactive
    iterations = arguments.iterations
    seed = arguments.seed
    config = arguments.config
    load_file = arguments.load_file
    explore = arguments.explore
    train = arguments.train
    render = arguments.render or interactive
    verbose = arguments.verbose or interactive
    global run_id
    run_id = arguments.run_id or run_id

    if seed:
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    if explore:
        print('Exploring Hyper-Parameters')
        explore_options(iterations, verbose)
        sys.exit()

    agent = setup(config, load_file)
    if train:
        print('Training...')
        print(f'Starting at: {time.strftime("%H:%M:%S")}')
        start = time.time()
        rewards, episode_length = agent.train(env, iterations, callback=print_progress, verbose=verbose)
        print(f'Duration of training was {time.time() - start}')
        name = f'training{iterations}_{run_id}'
        save_rewards(name, rewards, episode_length, verbose=verbose)
        if config:
            name = trim(config, '/', '.yaml')
        elif load_file:
            name = trim(load_file, '/', '.pt')
        else:
            name = f'training_{iterations}'
        agent.save(name, run_id)
        print('Evaluating...')
        rewards = agent.evaluate(env, 100, render, verbose, interactive)
        name = f'trial{iterations}_{run_id}'
        save_rewards(name, rewards, window=5, verbose=True)
    elif load_file:
        print('Evaluating...')
        rewards = agent.evaluate(env, iterations, render, verbose, interactive)
        load_file = trim(load_file, '/', '.pt')
        name = f'{load_file}_trial{iterations}'
        save_rewards(name, rewards, window=5, verbose=verbose)
    print("\n\n**DONE**")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lunar Lander')
    # verbosity
    parser.add_argument('-v', help='Turn on verbose tracing',
                        dest='verbose', const=True, default=False, action='store_const')
    # id
    parser.add_argument('--id', help='Specific run ID to make file identification easier.',
                        dest='run_id', type=str, default=run_id)
    # explore
    parser.add_argument('-e', '--explore', help='explore random hyper-parameters',
                        dest='explore', const=True, default=False, action='store_const')
    # config file
    parser.add_argument('-c', '--config', help='Agent is set up with given config file options',
                        dest='config', type=str)
    # train
    parser.add_argument('-t', '--train', help='Runs training before evaluation',
                        dest='train', const=True, default=False, action='store_const')
    # load file path
    parser.add_argument('-l', '--load', help='Path to the PyTorch (.pt) file to load',
                        dest='load_file', type=str)
    # iterations
    parser.add_argument('-i', '--iterations', help='Iterations to run or train for',
                        dest='iterations', default=100, type=int)
    # seed
    parser.add_argument('-s', '--seed', help='Seed the random number generators for (more) consistent results.'
                                             'Note that torch uses some non-deterministic behaviors for efficiency',
                        dest='seed', default=0, type=int)
    # render
    parser.add_argument('-r', '--render', help='Render each frame',
                        dest='render', const=True, default=False, action='store_const')
    # interactive
    parser.add_argument('--inter', help='Enter interactive mode when running agent',
                        dest='interactive', const=True, default=False, action='store_const')
    # parse
    args = parser.parse_args()
    print("Starting with arguments:", args)
    main(args)
