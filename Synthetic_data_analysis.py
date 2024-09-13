import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn.objects as so
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

df_words_similarities = pd.read_csv('words_similarities_semanthic.csv', index_col=0)



df_locations_similarities = pd.read_csv('locations_similarities_semanthic.csv', index_col=0)


df_env_states = pd.read_csv('synthetic_semantic.csv', index_col=0)



df_env_novel_states = pd.read_csv('synthetic_semantic_cue.csv', index_col=0)



############################################
import seaborn as sns
# Plot heatmaps for word similarities
plt.figure(figsize=(10, 8))
sns.heatmap(df_words_similarities.astype(float), annot=False, cmap="coolwarm", cbar=True)
plt.title('Word Similarities Heatmap')
#plt.show()
#df_words_similarities.to_csv('words_similarities_temporal_bias.csv')

df_locations_similarities.to_csv('locations_similarities_sanity_check.csv')

# Sample function to create DataFrame containing sequences
def create_df_sequences(seqs, df_env_states):
    df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
    for i, seq in enumerate(seqs):  # Iterate over each sequence
        df_i = df_env_states.iloc[seq]  # Select rows corresponding to states in the sequence
        df_i["seq"] = i  # Append a column with the sequence number
        df_sequences = df_sequences._append(df_i, ignore_index=True)  # Append the sequence to the DataFrame

        # Debugging output
        print(f"Trajectory {i}: Length = {len(df_i)}, States: {df_i['word'].tolist()}")

    return df_sequences




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


    env.plot_stochastic_matrix()

    # env = RoomWorld()

    # Create the generator
    generator = Generator(env)


    # Create the propagator
    propagator = Propagator(generator)
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




# Create a pandas DataFrame with the states sequence
df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
for model, seqs in seqs_score.items():
    df_sequences_i = create_df_sequences(seqs, df_env_states)
    df_sequences_i["model"] = model  # Add a column for the model name
    df_sequences = df_sequences._append(df_sequences_i, ignore_index=True)  # Append the sequences to the DataFrame





episodic_rl_label = "Episodic RL"
rl_thresh = 0.4
seqs_score[episodic_rl_label] = episodic_rl_algorithm(df_env_states, df_env_novel_states, df_words_similarities, max_seqs=n_step,
                                                      n_samp=n_samp, thresh=rl_thresh)



# Sample function to create DataFrame containing sequences
def create_df_sequences(seqs, df_env_states):
    df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
    for i, seq in enumerate(seqs):  # Iterate over each sequence
        df_i = df_env_states.iloc[seq]  # Select rows corresponding to states in the sequence
        df_i["seq"] = i  # Append a column with the sequence number
        df_sequences = df_sequences._append(df_i, ignore_index=True)  # Append the sequence to the DataFrame

        # Debugging output
        print(f"Trajectory {i}: Length = {len(df_i)}, States: {df_i['word'].tolist()}")

    return df_sequences




# Create a pandas DataFrame with the states sequence
df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
for model, seqs in seqs_score.items():
    df_sequences_i = create_df_sequences(seqs, df_env_states)
    df_sequences_i["model"] = model  # Add a column for the model name
    df_sequences = df_sequences._append(df_sequences_i, ignore_index=True)  # Append the sequences to the DataFrame



# Create a pandas DataFrame with the states sequence
df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
for model, seqs in seqs_score.items():
    df_sequences_i = create_df_sequences(seqs, df_env_states)
    df_sequences_i["model"] = model  # Add a column for the model name
    df_sequences = df_sequences._append(df_sequences_i, ignore_index=True)  # Append the sequences to the DataFrame

# Call the function with df_sequences DataFrame
#compute_evaluation_seq(df_sequences)

def compute_evaluation(df_sequences, df_env_novel_states):
    # Group_by model
    df_sequences_by_model = df_sequences

    # To each unique word of the model apply the semantic similarity to the novel words and get the mean
    df_sequences_by_model = df_sequences_by_model.groupby(['model', 'seq'])

    df_final_data = []  # List to accumulate data

    # iterate over the groups
    for seq_i, df_seq_i in df_sequences_by_model:
        # iterate over rows
        reward = 0
        # remove duplicated rows
        df_seq_i = df_seq_i.drop_duplicates(subset=['word', 'location'])
        for i, row in df_seq_i.iterrows():
            # iterate over the novel words
            for j, novel_row in df_env_novel_states.iterrows():
                # Get the semantic and spatial similarity
                semantic_sim = df_words_similarities.loc[row['word'], novel_row['word']]
                spatial_sim = df_locations_similarities.loc[row['location'], novel_row['location']]
                temporal_sim = 1-abs(row["time"] - novel_row["time"])/(max(row["time"],novel_row["time"]+ 1e-5))
                # Compute the reward
                similarity = (semantic_sim + spatial_sim + temporal_sim)/3

                reward += similarity
        #final_reward = reward/len(df_seq_i)

        # Append data to list
        df_final_data.append({'model': df_seq_i['model'].values[0], 'seq': seq_i, 'reward': reward})

    # Convert list of dictionaries to DataFrame
    df_final = pd.DataFrame(df_final_data)

    # Compute the mean reward for each seq_i
    df_sequences_by_model_sim = df_final.groupby(['model', 'seq']).mean().reset_index()

    return df_sequences_by_model_sim

df_eval = compute_evaluation(df_sequences, df_env_novel_states)



fig, ax = plt.subplots(figsize=(8, 5))
p = (
    so.Plot(df_eval, x="model", y="reward", color="model")
    .add(so.Dash(alpha=0.3), so.Agg())
    .add(so.Dots(), so.Jitter())
    .label(
        x="Model",
        y="Reward",
    )
    .layout(
        engine="tight",
    )
    .on(ax)
    .plot()
)
plt.show()
# save as pdf
fig.savefig("evaluation_sanity_check.pdf", bbox_inches="tight")



state_exploration_count = df_sequences.groupby(['model', 'word']).size().reset_index(name='counts')

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=state_exploration_count, x='word', y='counts', hue='model', ax=ax)
ax.set_title('State Exploration Count by Model')
ax.set_xlabel('State')
ax.set_ylabel('Count')
ax.legend(title='Model')
plt.xticks(rotation=90)
plt.show()
fig.savefig("state_exploration_count.pdf", bbox_inches="tight")



# Pivot the data to create a heatmap
state_exploration_pivot = df_sequences.pivot_table(index='word', columns='model', aggfunc='size', fill_value=0)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(state_exploration_pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
ax.set_title('State Exploration Heatmap')
ax.set_xlabel('Model')
ax.set_ylabel('State')
plt.xticks(rotation=90)
plt.show()
fig.savefig("state_exploration_heatmap.pdf", bbox_inches="tight")




# Create a transition matrix for each model
transition_matrices = {}
for model, seqs in seqs_score.items():
    transition_matrix = np.zeros((len(df_env_states), len(df_env_states)))
    for seq in seqs:
        for i in range(len(seq) - 1):
            if seq[i] != -1 and seq[i + 1] != -1:
                transition_matrix[seq[i], seq[i + 1]] += 1
    transition_matrices[model] = transition_matrix

for model, matrix in transition_matrices.items():
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu", ax=ax)
    ax.set_title(f'State Transition Heatmap for {model}')
    ax.set_xlabel('Next State')
    ax.set_ylabel('Previous State')
    plt.show()
    fig.savefig(f"state_transition_heatmap_{model.replace(' ', '_')}.pdf", bbox_inches="tight")



# Count the number of unique episodes visited by each model
episode_coverage = df_sequences.groupby(['model', 'episode']).size().reset_index(name='counts')
episode_coverage = episode_coverage.groupby('model').size().reset_index(name='episode_count')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=episode_coverage, x='model', y='episode_count', ax=ax)
ax.set_title('Episode Coverage by Model')
ax.set_xlabel('Model')
ax.set_ylabel('Number of Episodes Covered')
plt.xticks(rotation=45)
plt.show()
fig.savefig("episode_coverage.pdf", bbox_inches="tight")


# Count the number of unique states visited by each model
unique_state_visits = df_sequences.groupby(['model', 'word']).size().reset_index(name='counts')
unique_state_visits = unique_state_visits.groupby('model').size().reset_index(name='unique_state_count')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=unique_state_visits, x='model', y='unique_state_count', ax=ax)
ax.set_title('Unique State Visits by Model')
ax.set_xlabel('Model')
ax.set_ylabel('Number of Unique States Visited')
plt.xticks(rotation=45)
plt.show()
fig.savefig("unique_state_visits.pdf", bbox_inches="tight")


# Calculate unique state visits over time
df_sequences['time_step'] = df_sequences.groupby(['model', 'seq']).cumcount()
unique_states_over_time = df_sequences.groupby(['model', 'time_step'])['word'].nunique().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=unique_states_over_time, x='time_step', y='word', hue='model', ax=ax)
ax.set_title('Unique State Visits Over Time by Model')
ax.set_xlabel('Time Step')
ax.set_ylabel('Number of Unique States Visited')
plt.show()
fig.savefig("unique_state_visits_over_time.pdf", bbox_inches="tight")



def create_word_cooccurrence_graph(sequences, threshold=1):
    G = nx.Graph()
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] != -1 and seq[i + 1] != -1:
                if G.has_edge(seq[i], seq[i + 1]):
                    G[seq[i]][seq[i + 1]]['weight'] += 1
                else:
                    G.add_edge(seq[i], seq[i + 1], weight=1)

    # Remove edges with weights below threshold
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold])
    return G


# Create and plot word co-occurrence networks for each model
for model, seqs in seqs_score.items():
    G = create_word_cooccurrence_graph(np.vstack(seqs))
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold",
            edge_color="gray")
    plt.title(f'Word Co-occurrence Network for {model}')
    plt.show()
    plt.savefig(f"word_cooccurrence_network_{model.replace(' ', '_')}.pdf", bbox_inches="tight")


# Aggregate state distributions over time
state_distribution = df_sequences.groupby(['model', 'time_step', 'word']).size().reset_index(name='count')
state_distribution_pivot = state_distribution.pivot_table(index=['model', 'time_step'], columns='word', values='count', fill_value=0)

def compute_sequence_similarity(sequences):
    # Flatten the sequences into a 2D array where rows are sequences and columns are states
    flattened_seqs = [seq.flatten() for seq in sequences]
    return cosine_similarity(flattened_seqs)

# Compute similarities for each model's sequences
similarity_matrices = {}
for model, seqs in seqs_score.items():
    similarity_matrices[model] = compute_sequence_similarity(seqs)

# Plot the similarity matrix for each model
for model, matrix in similarity_matrices.items():
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu", ax=ax, annot=True, fmt=".2f")
    ax.set_title(f'Sequence Similarity Matrix for {model}')
    ax.set_xlabel('Sequence Index')
    ax.set_ylabel('Sequence Index')
    plt.show()
    fig.savefig(f"sequence_similarity_matrix_{model.replace(' ', '_')}.pdf", bbox_inches="tight")
