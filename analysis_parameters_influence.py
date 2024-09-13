import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn.objects as so
import networkx as nx

df_words_similarities = pd.read_csv('words_similarities_semanthic.csv', index_col=0)



df_locations_similarities = pd.read_csv('locations_similarities_semanthic.csv', index_col=0)


df_env_states = pd.read_csv('synthetic_semantic.csv', index_col=0)



df_env_novel_states = pd.read_csv('synthetic_semantic_cue.csv', index_col=0)




from Environment_Episodic import EpisodicGraph
from Generator_Episodic import Generator
from Propagator_Episodic import Propagator
from Simulator_Episodic import EpisodicSimulator
from episodic_replay_model import episodic_rl_algorithm


# Create the environment
k = 0.6  # Inter-episode connectivity. Domain: [0, 1]. 1, sample across episodes. 0, sample within episodes
m = 0  # Semantic Similarity weight. Domain: [0, inf]. 0, no action dependence.
n = 0  # Temporal Similarity weight. Domain: [0, inf]. 0, no time dependence.
o = 0  # Spatial Similarity weight. Domain: [0, inf]. 0, no spatial dependence.

models = {
    "Episodic Inference": (0, 0, 0),
    "Episodic Inference\nsemantic-biased": (1, 0, 0),
    "Episodic Inference\ntemporal-biased": (0, 1, 0),
    "Episodic Inference\nspatial-biased": (0, 0, 1),
}

init_state = 4
n_step = 10
n_samp = 20
seqs_score = {}
for model, params in models.items():
    k = 0.6
    m, n, o = params
    env = EpisodicGraph(df_env_states, df_words_similarities, df_locations_similarities, k=k, m=m, n=n, o=o)


    #env.plot_stochastic_matrix()

    # env = RoomWorld()

    # Create the generator
    generator = Generator(env)
    #genera
    #tor.plot_generator_matrix()

    # Create the propagator
    propagator = Propagator(generator)
    #propagator.plot_activation_matrix()
   #propagator.plot_et0_matrix()
    #propagator.plot_activation_matrix()

    # propagator.min_zero_cf()

    # Create the simulator

    # random init state
    init_state = 4
        #(np.random.randint(0, len(env.states)))

    simulator = EpisodicSimulator(propagator, init_state)

    simulator.sample_sequences(n_step=n_step, n_samp=n_samp)
    # Generate data for the GIF



    seqs_score[model] = simulator.state_seqs
