import numpy as np


class AgentBuilder:
    def __init__(self, num_inputs, num_actions):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.agent = None
        self.current_config = None
        self.config_records = []

        self.rars = [0.95, 1.0, 1.05, 1.1, 1.2]
        self.rar_decays = [0.99996, 0.99997, 0.99998, 0.99999]
        self.minibatch_sizes = [128, 256, 512]
        self.memories = [100000, 120000, 130000, 150000, 170000]
        self.replay_freqs = [10, 20, 30, 40, 50]
        self.target_updates = [80000, 100000, 125000]
        self.gammas = [0.9992, 0.9995, 0.9998]
        self.features = ['gammas', 'rars', 'rar_decays', 'memories', 'minibatch_sizes',
                          'replay_freqs', 'target_updates']


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
        config = {'gamma': 0.9992, 'rar': 1.0, 'rar_decay': 0.99999,
                  'memory_size': 250000, 'minibatch_size': 512,
                  'replay_freq': 50, 'target_update': 100000}
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
                for feature, value in config.iteritems():
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
        config = { 'gamma':gamma, 'rar': rar, 'rar_decay': rar_decay,
                   'memory_size': memory, 'minibatch_size': minibatch_size,
                   'replay_freq': replay_freq, 'target_update': target_update}
        return config

    # def __iter__(self, iterations=100):
    #     self.iterations = iterations
    #     return self

    def next(self):
        # if not self.iterations:
        #     raise StopIteration
        # else:
        # self.iterations -= 1
        return self._explore()




class ModelBuilder:
    def __init__(self, num_inputs, num_actions):
        self.possible_units = [[128, 256], [32, 16]]
        self.learning_rates = [0.0005, 0.001, 0.0015, 0.002]
        self.num_inputs = num_inputs
        self.num_actions = num_actions

    @staticmethod
    def build(units=(16, 72), lr=0.005, num_inputs=8, num_actions=4):
        """
        Build a new NN model
        :param num_actions: Number of actions to output for the last layer
        :param num_inputs: Number of inputs to expect on the first layer
        :param units: A list of units to be used in each layer.
        :type units: list
        :param lr: Learning Rate
        :return Sequential: A new NN keras Sequential model
        """
        model = Sequential()
        model.add(Dense(units[0], input_dim=num_inputs, activation='relu'))
        # model.add(Activation('relu'))  # I hear RELU this is a standard for NNs these days
        if units[1]:
            model.add(Dense(units[1], activation='relu'))
        # model.add(Activation('relu'))
        model.add(Dense(num_actions, activation='linear'))
        # model.add(Activation('linear'))  # numpy will pick the argmax from these values
        # logcosh as it is most similar to Huber loss mentioned in DeepMind video
        model.compile(loss='mse', optimizer='adam')
        model.learning_rate = lr
        return model


class RandomModelBuilder(ModelBuilder):
    def __init__(self, num_inputs, num_actions):
        ModelBuilder.__init__(self, num_inputs, num_actions)
        self.iterations = 100

    def _explore(self):
        units = [int(np.random.choice(self.possible_units[0], 1)),
                 int(np.random.choice(self.possible_units[1], 1))]
        learning_rate = float(np.random.choice(self.learning_rates, 1))
        return ModelBuilder.build(units, learning_rate, self.num_inputs, self.num_actions)


    # def __iter__(self, iterations):
    #     self.iterations = iterations
    #     return self

    def next(self):
        # if not self.iterations:
        #     raise StopIteration
        # else:
        #     self.iterations -= 1
        return self._explore()