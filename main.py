import sys
import argparse
import time
import gym
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from agents import DQNAgent
from auto import Runner
from builders import AgentLoader

env = gym.make('LunarLander-v2')
# env = gym.make('CartPole-v0')
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n


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


def print_progress(agent, data):
    percent = data['percent']
    if percent % 5 != 0:
        return
    progress = '=' * int(percent)
    progress += '>'
    left = ' ' * (100 - percent)
    print(f'{percent}% [{progress + left}]')
    reward, steps = data['stats']
    total = round(reward.mean(), 3)
    last100 = round(reward[-100:].mean(), 3)
    steps = steps.sum()
    print(f'Total Mean: {total},  Last 100 Mean: {last100},  Steps: {steps}')


def moving_average(array, n=20):
    ret = np.cumsum(array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[(n - 1):] / n


def save_rewards(filename, rewards):
    plt.clf()
    plt.plot(rewards, 'bo', label='Episode Reward')
    plt.plot(moving_average(rewards), 'r', label='Rolling Mean (20)')
    plt.axhline(200, 0, 1, color='g')
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episode')
    plt.savefig(f'figures/{filename}.png')


def trim(s: str, start: str, end: str) -> str:
    return s[(s.rfind(start)+1):s.rfind(end)]


def main(arguments):
    interactive = arguments.interactive
    iterations = arguments.iterations
    config = arguments.config
    load_file = arguments.load_file
    explore = arguments.explore
    train = arguments.train
    render = arguments.render or interactive
    verbose = arguments.verbose or interactive

    if explore:
        print('Exploring Hyper-Parameters')
        explore_options(iterations, verbose)
        sys.exit()

    agent = setup(config, load_file)
    if train:
        print('Training...')
        rewards, episode_length = agent.train(env, iterations, callback=print_progress)
        name = f'training{iterations}_{time.strftime("%H-%M-%S")}'
        save_rewards(name, rewards)
        if config:
            name = trim(config, '/', '.yaml')
        elif load_file:
            name = trim(load_file, '/', '.pt')
        else:
            name = f'training_{iterations}'
        agent.save(name, time.strftime("%H-%M-%S"))
        print('Evaluating...')
        rewards = agent.evaluate(env, 100, render, verbose, interactive)
        name = f'trial{iterations}_{time.strftime("%H-%M-%S")}'
        save_rewards(name, rewards)
    elif load_file:
        print('Evaluating...')
        rewards = agent.evaluate(env, iterations, render, verbose, interactive)
        load_file = trim(load_file, '/', '.pt')
        name = f'{load_file}_trial{iterations}'
        save_rewards(name, rewards)
    print("\n\n**DONE**")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lunar Lander')
    # verbosity
    parser.add_argument('-v', help='Turn on verbose tracing',
                        dest='verbose', const=True, default=False, action='store_const')
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
