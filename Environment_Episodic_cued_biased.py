from Environment_Episodic import EpisodicGraph
from utils import row_norm
import networkx as nx
import numpy as np
import pandas as pd



class EpisodicGraph_cued_biasy(EpisodicGraph):
    def __init__(self, states, semantic_sim, spatial_sim, cue_states, k, m, n, o, p, start=4):
        self.n_state = len(states)
        self.start = start
        self.states = states
        self.semantic_sim = semantic_sim
        self.spatial_sim = spatial_sim
        self.cue_states = cue_states
        self.k = k
        self.m = m
        self.n = n
        self.o = o
        self.p = p
        self._access_matrix()
        self._state_information()
        self.__name__ = "EpisodicGraph_cued-biasy"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.T_torch = None
        self.fname_graph = "figures/episodic_graph.png"




    def _access_matrix(self):
        """
        Sets the adjacency/stochastic matrix for the community graph.
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        """

        def compute_sequence_similarity(state, sequence_states, semantic_sim, spatial_sim):
            """
            Computes the similarity between a given state and a sequence of states.
            The similarity is based on semantic, temporal, and spatial attributes.

            Args:
                state (pd.Series): The state to compare.
                sequence_states (pd.DataFrame): The sequence of states to compare against.
                semantic_sim (pd.DataFrame or float): Semantic similarity matrix or a fixed similarity value.
                spatial_sim (pd.DataFrame or float): Spatial similarity matrix or a fixed similarity value.

            Returns:
                float: The similarity score.
            """
            total_similarity = 0
            num_sequence_states = len(sequence_states)

            for _, seq_state in sequence_states.iterrows():
                # Extracting the word from seq_state
                word_seq = seq_state['word']

                # Extracting semantic similarity
                if isinstance(semantic_sim, pd.DataFrame):
                    try:
                        semantic_s = semantic_sim.at[state['word'], word_seq]
                    except KeyError as e:
                        print(f"KeyError in semantic_sim access: {state['word']}, {word_seq}")
                        raise
                else:
                    semantic_s = semantic_sim
                semantic_s = float(semantic_s)

                temporal_s = 1 - abs(state['time'] - seq_state['time']) / (max(state['time'], seq_state['time']) + 1e-5)
                episodic_s = 1 - abs(state['episode'] - seq_state['episode']) / (max(state['episode'], seq_state['episode']) + 1e-5)


                # Extracting spatial similarity
                if isinstance(spatial_sim, pd.DataFrame):
                    try:
                        spatial_s = spatial_sim.loc[state['location'], seq_state['location']]
                    except KeyError as e:
                        print(f"KeyError in spatial_sim access: {state['location']}, {seq_state['location']}")
                        raise
                else:
                    spatial_s = spatial_sim
                spatial_s = float(spatial_s)

                similarity = (semantic_s + temporal_s + spatial_s + episodic_s) / 4
                total_similarity += similarity

            average_similarity = total_similarity / num_sequence_states
            return average_similarity
        def compute_v(state_i, state_j, semantic_sim, spatial_sim, cue_states, k, m, n, o, p):
            # Extracting the word from state_j
            word_j = state_j['word'].iloc[0] if isinstance(state_j['word'], pd.Series) else state_j['word']

            # Extracting semantic similarity
            if isinstance(semantic_sim, pd.DataFrame):
                try:
                    semantic_s = semantic_sim.at[state_i['word'], word_j]
                except KeyError as e:
                    print(f"KeyError in semantic_sim access: {state_i['word']}, {word_j}")
                    raise
            else:
                semantic_s = semantic_sim
            semantic_s = float(semantic_s)

            temporal_s = 1 - abs(state_i['time'] - state_j['time']) / (max(state_i['time'], state_j['time']) + 1e-5)

            # Extracting spatial similarity
            if isinstance(spatial_sim, pd.DataFrame):
                try:
                    spatial_s = spatial_sim.loc[state_i['location'], state_j['location']]
                except KeyError as e:
                    print(f"KeyError in spatial_sim access: {state_i['location']}, {state_j['location']}")
                    raise
            else:
                spatial_s = spatial_sim
            spatial_s = float(spatial_s)

            delta = 0 if state_i['episode'] == state_j['episode'] else 1

            # Compute sequence similarity
            sequence_similarity = compute_sequence_similarity(state_j, cue_states, semantic_sim, spatial_sim)

            V = (1.0001-k) ** (-delta) * (1.0001-semantic_s) ** (-m) * (1.0001-temporal_s) ** (-n) * (1.0001-spatial_s) ** (-o)* ( 1.0001- sequence_similarity) ** (-p)
            #V = (0.1+ k) ** (-delta) * (0.1+ semantic_s) ** (-m) * (0.1+ temporal_s) ** (-n) * (0.1+ spatial_s) ** (
             #   -o) * (0.1 +
              #         sequence_similarity) ** (-p)

            #V = V ** (1/((delta + m + n + o + p+ 0.0001)))
            #V= V/0.1**(-delta-m-n-o-p)
            #V = (k**delta +semantic_s**m +temporal_s**n +spatial_s**o + sequence_similarity**p)/(k+n+m+o+p +0.01)

            return V

        num_states = len(self.states)
        O = np.zeros((num_states, num_states))

        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    state_i = self.states.iloc[i]
                    state_j = self.states.iloc[j]
                    O[i, j] = compute_v(state_i, state_j, self.semantic_sim, self.spatial_sim,
                                        self.cue_states, self.k, self.m, self.n, self.o, self.p)

        for i in range(num_states):
            O[i, i] = -O[i, :].sum()

        self.A = np.zeros((num_states, num_states))
        n = -np.diag(O)

        if np.all(n == 0):
            print("Warning: All diagonal elements are zero, resulting in a zero adjacency matrix A.")
        else:
            for i in range(num_states):
                for j in range(num_states):
                    if i != j:
                        if n[i] != 0:  # Avoid division by zero
                            self.A[i, j] = O[i, j] / n[i]
                        else:
                            self.A[i, j] = 0
                    else:
                        self.A[i, j] = 0

        # Construct the stochastic matrix T from A and normalize the rows
        self.T = self.A.copy()
        for i in range(num_states):
            row_sum = np.sum(self.T[i, :])
            if not np.isclose(row_sum, 0):  # Avoid normalization for zero rows
                self.T[i, :] /= row_sum