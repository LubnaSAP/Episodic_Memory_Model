from environments import GraphEnv
from utils import row_norm
import networkx as nx
import numpy as np
import pandas as pd


class EpisodicGraph(GraphEnv):
    def __init__(self, states, semantic_sim, spatial_sim, k=0.1, m=1, n=1, o=1, start=0):
        self.n_state = len(states)
        self.start = start
        self.states = states
        self.semantic_sim = semantic_sim
        self.spatial_sim = spatial_sim
        self.k = k
        self.m = m
        self.n = n
        self.o = o
        self._access_matrix()
        super(EpisodicGraph, self).__init__()
        self._state_information()
        self.__name__ = "EpisodicGraph"
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

        def compute_v(state_i, state_j, semantic_sim, spatial_sim, k, m, n, o):
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

            temporal_s = 1 - abs(state_i['time'] - state_j['time'])

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

            #V = (k) ** delta*(semantic_s) ** m *(temporal_s) ** n * (spatial_s) **o

            V = (1.0001-k) ** (-delta) * (1.0001-semantic_s) ** (-m) * (1.0001-temporal_s) ** (-n) * (1.0001-spatial_s) ** (-o)
            #V = (1.1 - k) ** (-delta) * (semantic_s) ** (m) * (1.1 - temporal_s) ** (-n) * (1.1 - spatial_s) ** (-o)
            #V = (0.1+ k) ** (-delta) * (0.1+ semantic_s) ** (-m) * (0.1+ temporal_s) ** (-n) * (0.1+ spatial_s) ** (-o)
            #V = (k * delta + semantic_s * m + temporal_s * n + spatial_s * o) #/ (
                        #k + n + m + o +0.01 )


            V = V ** (1/(-(delta + m + n + o +0.01)))

            return V

        num_states = len(self.states)
        O = np.zeros((num_states, num_states))

        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    state_i = self.states.iloc[i]
                    state_j = self.states.iloc[j]
                    O[i, j] = compute_v(state_i, state_j, self.semantic_sim, self.spatial_sim, self.k, self.m, self.n, self.o)

        # Verify the matrix O before adjusting the diagonal
        #print("Matrix O before adjusting the diagonal:")
        #print(O)

        for i in range(num_states):
            O[i, i] = -O[i, :].sum()

        # Verify the diagonal of O after adjusting
        #print("Matrix O after adjusting the diagonal:")
        #print(O)

        self.A = np.zeros((num_states, num_states))
        n = -np.diag(O)

        # Verify the vector n
        #print("Vector n:")
        #print(n)

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

        # Verify the adjacency matrix A
        #print("Matrix A:")
        #print(self.A)

        # Construct the stochastic matrix T from A and normalize the rows
        self.T = self.A.copy()
        for i in range(num_states):
            row_sum = np.sum(self.T[i, :])
            if not np.isclose(row_sum, 0):  # Avoid normalization for zero rows
                self.T[i, :] /= row_sum

        # Verify the stochastic matrix T
        #print("Matrix T:")
        #print(self.T)


    @property
    def node_layout(self):
        """Return node_layout."""
        return self._node_layout

    def _set_graph(self):
        """Defines networkx graph including info_state information"""
        nodesdf = self.info_state.reset_index()
        edgesdf = self.info_transition
        G = nx.from_pandas_edgelist(
            df=edgesdf, source="source", target="target", edge_attr="weight"
        )
        nx.set_node_attributes(G, name="x", values=nodesdf.x.to_dict())
        nx.set_node_attributes(G, name="y", values=nodesdf.y.to_dict())
        self.G = G

    def _compute_reward(self, state, novel_episode: pd.DataFrame):
        """
        Computes reward for a given state based on transitions and similarities without weights.

        INPUTS:
            state - current state (dict with 'word', 'location', 'time')
            novel_episode - DataFrame of novel states

        OUTPUT:
            reward - computed reward score
        """
        total_reward = 0
        count = 0

        # Ensure the semantic and spatial similarity matrices are valid
        if not isinstance(self.semantic_sim, pd.DataFrame) or not isinstance(self.spatial_sim, pd.DataFrame):
            raise ValueError("Semantic and Spatial similarity should be pandas DataFrames")

        for i, row in novel_episode.iterrows():
            # Semantic similarity
            if state["word"] in self.semantic_sim.index and row["word"] in self.semantic_sim.columns:
                sem_similarity = self.semantic_sim.loc[state["word"], row["word"]]
            else:
                sem_similarity = 0  # Set to 0 if no valid similarity is found

            # Spatial similarity
            if state["location"] in self.spatial_sim.index and row["location"] in self.spatial_sim.columns:
                spa_similarity = self.spatial_sim.loc[state["location"], row["location"]]
            else:
                spa_similarity = 0  # Set to 0 if no valid similarity is found

            # Temporal similarity
            time_difference = 1 - abs(state["time"] - row["time"])  # Normalize to [0, 1]

            # Aggregate rewards
            total_reward += sem_similarity + spa_similarity + time_difference
            count += 1

        # Normalize reward by the number of comparisons
        if count > 0:
            reward = total_reward / count
        else:
            reward = 0

        return reward
