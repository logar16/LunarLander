import numpy as np
import torch.nn as nn
import torch.optim as optim
import yaml


class AgentBuilder:
    def __init__(self, num_inputs, num_actions):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.agent = None
        self.current_config = None
        self.config_records = []

        self.rars = [0.99, 1.0, 1.05, 1.1, 1.2]
        self.rar_decays = [0.99996, 0.99997, 0.99998, 0.99999]
        self.minibatch_sizes = [128, 256, 512]
        self.memories = [150000, 200000, 300000, 400000, 500000]
        self.replay_freqs = [10, 20, 30, 40, 50]
        self.target_updates = [50000, 80000, 100000, 120000]
        self.gammas = [0.9994, 0.9996, 0.9998, 0.9999, 0.99995]
        self.learning_rates = [0.00025, 0.0005, 0.001, 0.0015, 0.002]
        self.optim_types = [optim.Adam, optim.AdamW, optim.Adadelta]  #, optim.ASGD, optim.SGD
        self.features = ['gammas', 'rars', 'rar_decays', 'memories', 'minibatch_sizes',
                         'replay_freqs', 'target_updates', 'optim_type', 'optim_args']

    def create_agent(self, config):
        from agents import DQNAgent
        agent = DQNAgent(num_actions=self.num_actions, num_inputs=self.num_inputs, config=config, **config)
        self.agent = agent
        self.current_config = config
        return agent
        # Anything else?


class SpecificAgentBuilder(AgentBuilder):
    def __init__(self, num_inputs, num_actions):
        AgentBuilder.__init__(self, num_inputs, num_actions)

    def next(self):
        optim_args = {'lr': 0.005}
        config = {
            'gamma': 0.9992,
            'rar': 1.0, 'rar_decay': 0.99999,
            'memory_size': 250000, 'minibatch_size': 512,
            'replay_freq': 50, 'target_update': 100000,
            'optim_type': optim.SGD, 'optim_args': optim_args,
            'criterion': nn.SmoothL1Loss()
        }
        agent = self.create_agent(config)
        # agent.load('Files/nn_65.37.h5')
        return agent


class RandomAgentBuilder(AgentBuilder):
    def __init__(self, num_inputs, num_actions):
        AgentBuilder.__init__(self, num_inputs, num_actions)
        self.iterations = 100

    def _explore(self):
        """Pick random features to find what works best generally.  Later zoom in"""
        new_enough = False
        max_matches = 3
        config = None
        while not new_enough:
            config = self._randomly_select_features()
            new_enough = True
            for other in self.config_records:
                matches = 0
                for feature, value in config.items():
                    if other[feature] == value:
                        matches += 1
                if matches >= max_matches:
                    new_enough = False
                    break
        self.config_records.append(config)
        return self.create_agent(config)

    def _randomly_select_features(self):
        gamma = np.random.choice(self.gammas)
        rar = np.random.choice(self.rars)
        rar_decay = np.random.choice(self.rar_decays)
        memory = np.random.choice(self.memories)
        minibatch_size = np.random.choice(self.minibatch_sizes)
        replay_freq = np.random.choice(self.replay_freqs)
        target_update = np.random.choice(self.target_updates)

        optim_type, optim_args = self.pick_optimizer()

        config = {
            'gamma': gamma, 'rar': rar, 'rar_decay': rar_decay,
            'memory_size': memory, 'minibatch_size': minibatch_size,
            'replay_freq': replay_freq, 'target_update': target_update,
            'optim_type': optim_type, 'optim_args': optim_args,
        }
        return config

    # def __iter__(self, iterations=100):
    #     self.iterations = iterations
    #     return self

    def _pick_optimizer(self):
        optim_type = np.random.choice(self.optim_types, 1)[0]
        lr = float(np.random.choice(self.learning_rates, 1))
        optim_args = {'lr': lr}
        if optim_type == optim.SGD:
            optim_args['momentum'] = 0.5
            # optim_args['weight_decay'] = 0.1
        elif optim_type == optim.Adadelta:
            optim_args['lr'] = 1.0
        # elif optim_type == optim.ASGD:
        #     optim_args['weight_decay'] = 0.1
        # elif optim_type == optim.Adagrad:
        #     optim_args['lr_decay'] = np.random.rand() * 0.000001
        return optim_type, optim_args

    def next(self):
        # if not self.iterations:
        #     raise StopIteration
        # else:
        # self.iterations -= 1
        return self._explore()


class ModelBuilder:
    def __init__(self, num_inputs, num_actions):
        self.possible_units = [[32, 64, 128, 256], [32, 16, 8]]
        self.num_inputs = num_inputs
        self.num_actions = num_actions

    @staticmethod
    def build(units=(128, 32), num_inputs=8, num_actions=4):
        """
        Build a new NN model
        :param num_actions: Number of actions to output for the last layer
        :param num_inputs: Number of inputs to expect on the first layer
        :param units: A list of units to be used in each layer.
        :type units: list
        :return Module: A new  model
        """
        # TODO: CUDA
        layers = [nn.Linear(num_inputs, units[0]), nn.ReLU()]
        if units[1]:
            layers.append(nn.Linear(units[0], units[1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(units[-1], num_actions))
        model = nn.Sequential(*layers)
        return model


class RandomModelBuilder(ModelBuilder):
    def __init__(self, num_inputs, num_actions):
        ModelBuilder.__init__(self, num_inputs, num_actions)
        self.iterations = 100

    def _explore(self):
        units = [int(np.random.choice(self.possible_units[0], 1)),
                 int(np.random.choice(self.possible_units[1], 1))]
        return ModelBuilder.build(units, self.num_inputs, self.num_actions)

    # def __iter__(self, iterations):
    #     self.iterations = iterations
    #     return self

    def next(self):
        # if not self.iterations:
        #     raise StopIteration
        # else:
        #     self.iterations -= 1
        return self._explore()


class AgentLoader(AgentBuilder):
    def __init__(self, path, num_inputs, num_actions):
        super().__init__(num_inputs, num_actions)
        self.path: str = path
        self.optimizers = {}
        for opt in self.optim_types:
            name = str(opt)
            start = name.find("'") + 1
            end = name.rfind("'")
            name = name[start:end]
            self.optimizers[name] = opt

    def load(self):
        """
        Note that `optim_type` in the config will need to match the name produced
            if the class is converted using str()
        :return:
        """
        print("Loading Config...")
        with open(self.path, 'r') as file:
            config = yaml.load(file)
            print(config)
            agent_config = config['agent']
            model_config = config['model']
            agent_config['optim_type'] = self.optimizers[agent_config['optim_type']]
            agent = self.create_agent(agent_config)
            model = ModelBuilder.build(model_config['units'], self.num_inputs, self.num_actions)
            agent.set_model(model)
            return agent
