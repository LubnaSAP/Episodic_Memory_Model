import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define the SimpleGenerator class with a method to plot the stochastic matrix
class SimpleGenerator:
    def __init__(self, env):
        self.env = env
        self.T =np.ones((self.env.n_state, self.env.n_state)) / self.env.n_state  # Uniform transition probabilities
        #self.T = env.T # Instead of choosing a uniform transition probabilities, the actual distribution for each model
        # based on the parameters k,m,n,o could be chosen to perfrom the non-diffsuive model

    def generate_sequence(self, init_state, n_step):
        sequence = [init_state]
        for _ in range(n_step - 1):
            current_state_index = sequence[-1]
            next_state_index = np.random.choice(
                range(self.env.n_state),  # Use indices to select the next state
                p=self.T[current_state_index]  # Uniform transition probabilities
            )
            sequence.append(next_state_index)
        return sequence


# Define the SimpleSimulator class
class SimpleSimulator:
    def __init__(self, generator, init_state):
        self.generator = generator
        self.init_state = init_state
        self.state_seqs = []

    def sample_sequences(self, n_step, n_samp):
        self.state_seqs = [self.generator.generate_sequence(self.init_state, n_step) for _ in range(n_samp)]


