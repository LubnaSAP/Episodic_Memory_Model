import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from Environment_Episodic import EpisodicGraph
from Generator_Episodic import Generator
from Propagator_Episodic import Propagator
from Simulator_Episodic import EpisodicSimulator
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import json

# definition of functions
def plot_environment(df_env_states, words_sorted, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    colors = ["r", "b"] * (len(df_env_states["episode"].unique()) // 2 + 1)

    from matplotlib.patches import Rectangle

    rectangle_height = len(df_env_states["word"].unique())
    i_prev = 0
    for c, i in zip(colors, [10] * len(df_env_states["episode"].unique())):
        rect = Rectangle((i_prev - 0.5, - 0.5), i, rectangle_height + 0.5, linewidth=1, facecolor=c, alpha=0.05)
        ax.add_patch(rect)
        # Plot locations name at the center of the rectangle
        # label_i = df_env_states.loc[i_prev]["episode"]
        if i_prev in df_env_states.index:
            label_i = df_env_states.loc[i_prev]["episode"]

        label = f"Ep. {label_i}"
        ax.text(i_prev + i / 2, rectangle_height - 3, label, fontsize=9, ha='center', alpha=0.5, color=c)

        i_prev += i

    # set limits
    ax.set_xlim(-1, len(df_env_states))
    ax.set_ylim(len(df_env_states["word"].unique()) - 0.5, -0.5)

    for i, row in df_env_states.iterrows():
        ax.scatter(row['time'] * (len(df_env_states) - 1), words_sorted[row['word']], color='k', s=50, alpha=0.2)

    # Shuffle y axis
    ax.set_yticks(np.arange(len(words_sorted.keys())))
    # Set words_sampled in reverse order
    ax.set_yticklabels(words_sorted.keys())

    return ax


def episodic_rl_algorithm(df_env_states, df_env_novel_states, max_seqs=4, n_samp=4, thresh=0.4):
    seqs = np.zeros((n_samp, max_seqs), dtype=int)

    for i in range(n_samp):
        seqs_i = []

        # Group df_env_states by location
        df_env_states_2 = df_env_states.copy()

        # Add a id column to df_env_states_2
        df_env_states_2['id'] = np.arange(len(df_env_states_2))
        df_env_states_by_location = df_env_states_2.groupby('episode')

        random_group_id = df_env_states['episode'].min()

        # Iterate until you reach the maximum episode ID
        while len(seqs_i) < max_seqs and random_group_id <= df_env_states['episode'].max():
            if random_group_id in df_env_states_by_location.groups:
                random_group = df_env_states_by_location.get_group(random_group_id)
                # Rest of your code here...
            random_group_id += 1  # Move to the next episode ID

            # Get random_row
            random_row = random_group.sample(n=1)
            seqs_i.append(random_row['id'].values[0])

            # Eval each word from df_env_novel_states with the random_row
            sim_max = 0
            for word in df_env_novel_states['word'].unique():
                # Get the similarity between the word and the random_row
                sim = df_words_similarities.loc[word, random_row['word'].values[0]]
                if sim > sim_max:
                    sim_max = sim

            if sim_max > thresh:
                random_rows = random_group.sample(frac=0.9, replace=False)
                [seqs_i.append(i) for i in random_rows['id'].values]

        seqs[i] = np.array(seqs_i)[:max_seqs]
    return seqs


def create_df_sequences(seqs):
    df_sequences = pd.DataFrame()
    for i in range(seqs.shape[0]):
        df_i = df_env_states.iloc[seqs[i, :]]
        # Append a column with the sequence number
        df_i["seq"] = i
        df_sequences = pd.concat([df_sequences, df_i], ignore_index=True)
    return df_sequences


# compute the evaluation
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
                # Compute the reward
                reward += semantic_sim + spatial_sim

        # Append data to list
        df_final_data.append({'model': df_seq_i['model'].values[0], 'seq': seq_i, 'reward': reward})

    # Convert list of dictionaries to DataFrame
    df_final = pd.DataFrame(df_final_data)

    # Compute the mean reward for each seq_i
    df_sequences_by_model_sim = df_final.groupby(['model', 'seq']).mean().reset_index()

    return df_sequences_by_model_sim


def compute_evaluation_model_participant(df_sequences, df_env_states_participants, df_words_similarities,
                                         df_locations_similarities, participant_numbers):
    df_combined_similarity_result = pd.DataFrame()  # DataFrame to store combined similarity scores

    for participant_num in participant_numbers:
        # Group by model and sequence
        df_sequences_by_model = df_sequences.groupby(['model', 'seq'])

        for (model, seq), df_seq_i in df_sequences_by_model:
            # Remove duplicated rows
            df_seq_i = df_seq_i.drop_duplicates(subset=['word', 'location'])

            similarity = 0  # Initialize similarity for the current sequence

            # Iterate over the rows of the current sequence
            for _, row in df_seq_i.iterrows():
                # Iterate over the rows of the participant data
                for _, participant_row in df_env_states_participants[participant_num].iterrows():
                    # Get the semantic and spatial similarity between the model's word/location and participant's word/location
                    semantic_sim = df_words_similarities.loc[row['word'], participant_row['word']]
                    spatial_sim = df_locations_similarities.loc[row['location'], participant_row['location']]
                    # Compute the similarity
                    similarity += semantic_sim + spatial_sim

            # Append data to the DataFrame
            df_combined_similarity_result = pd.concat([df_combined_similarity_result, pd.DataFrame(
                {'model': model, 'seq': seq, 'similarity': similarity, 'participant': participant_num}, index=[0])],
                                                      ignore_index=True)

    # Compute mean similarity separately for each model
    df_combined_similarity_mean = df_combined_similarity_result.groupby(['model', 'participant']).mean().reset_index()
    df_combined_similarity_mean_all = df_combined_similarity_mean.groupby(['model']).mean().reset_index()

    model_name = "Episodic Inference"

    final_mean_similarity_episodic_inference = df_combined_similarity_mean_all[
        df_combined_similarity_mean_all['model'] == model_name]
    return df_combined_similarity_result, df_combined_similarity_mean, df_combined_similarity_mean_all, \
        final_mean_similarity_episodic_inference['similarity']


# Load necessary DataFrames from the updated file paths
df_words_similarities = pd.read_csv('words_similarities_salience.csv', index_col=0)
df_locations_similarities = pd.read_csv('locations_similarities_salience.csv', index_col=0)
df_env_states = pd.read_csv('environment_salience.csv', index_col=0)
df_env_novel_states = pd.read_csv('environment_salience_novel.csv', index_col=0)

df_words_similarities_tuned = df_words_similarities

# Perform MDS on word similarities
mds = MDS(n_components=1, dissimilarity='precomputed', random_state=0)
semantic_mds = mds.fit_transform(df_words_similarities_tuned).reshape(-1)

# Perform MDS on spatial similarities
spatial_mds = mds.fit_transform(df_locations_similarities).reshape(-1)

# Scale time to be between 0 and 1
df_env_states['time'] = df_env_states['time'] / df_env_states['time'].max()

# Create DataFrames for semantic and spatial MDS results
df_semantic_mds = pd.DataFrame({'semantic_mds': semantic_mds}, index=df_words_similarities_tuned.index)
df_spatial_mds = pd.DataFrame({'spatial_mds': spatial_mds}, index=df_locations_similarities.index)

# Define participant numbers
participant_numbers = [1, 2, 3, 4, 13]

# Create an empty dictionary to store filtered DataFrames
df_env_states_participants = {}

# Iterate over each participant
for participant_num in participant_numbers:
    # Extract salience column for the current participant
    salience_column = f"salience_{participant_num}"

    # Select rows where salience is equal to 1
    df_participant = df_env_states[['word', 'location', 'time', 'episode']][df_env_states[salience_column] == 1]

    # Calculate the percentage of words with salience = 1
    total_words = len(df_env_states['word'].unique())
    words_with_salience = len(df_participant['word'].unique())
    percentage = (words_with_salience / total_words) * 100

    # Store the DataFrame in the dictionary
    df_env_states_participants[participant_num] = df_participant

    # Print the percentage
    print(f"Participant {participant_num}: {percentage}% of words have salience = 1")


# print(df_env_states_participants[2])

######################################################################################################################################


##########################################################################################################################################################
######################################################################


def create_environment(k, m, n, o):
    """Create the environment with given parameters."""
    env = EpisodicGraph(df_env_states, df_words_similarities_tuned,
                        df_locations_similarities, k=k, m=m, n=n, o=o)
    return env


def simulate_sequences(env, n_step, n_samp):
    """Simulate sequences in the environment."""
    generator = Generator(env)
    propagator = Propagator(generator)
    init_state = np.random.randint(0, len(env.states))
    simulator = EpisodicSimulator(propagator, init_state)
    simulator.sample_sequences(n_step=n_step, n_samp=n_samp)
    return simulator.state_seqs


def compute_evaluation(df_sequences, df_env_states_participants, df_words_similarities, df_locations_similarities,
                       participant_numbers):
    """Compute evaluation metrics."""
    df_combined_similarity_result = pd.DataFrame()
    for participant_num in participant_numbers:
        for (model, seq), df_seq_i in df_sequences.groupby(['model', 'seq']):
            df_seq_i = df_seq_i.drop_duplicates(subset=['word', 'location'])
            similarity = 0
            participant_autosimilarity = 0


            # for _, row_i in df_env_states_participants[participant_num].iterrows():
            #
            #     for _, participant_row in df_env_states_participants[participant_num].iterrows():
            #         participant_row_elem = len(df_env_states_participants[participant_num])
            #
            #         semantic_autosim = df_words_similarities.loc[row_i['word'], participant_row['word']]
            #         spatial_autosim = df_locations_similarities.loc[row_i['location'], participant_row['location']]
            #         participant_autosimilarity += semantic_autosim + spatial_autosim
            # participant_autosimilarity /= participant_row_elem**3

            for _, row in df_seq_i.iterrows():
                print('length of model sequence:', len(df_seq_i))
                best_semantic_sim, best_spatial_sim = 0, 0
                for _, participant_row in df_env_states_participants[participant_num].iterrows():
                    participant_row_elements = len(df_env_states_participants[participant_num])
                    #print(row['word'])
                    semantic_sim = df_words_similarities.loc[row['word'], participant_row['word']]
                    spatial_sim = df_locations_similarities.loc[row['location'], participant_row['location']]
                    #print(semantic_sim), print(spatial_sim)
                    best_semantic_sim = max(semantic_sim, best_semantic_sim)
                    best_spatial_sim = max(spatial_sim, best_spatial_sim)

                assert best_semantic_sim <= 1.00000001 and best_spatial_sim <=1.0000001,\
                    f'similarity too big {best_semantic_sim, best_spatial_sim}'
                similarity += best_semantic_sim + best_spatial_sim

            #similarity /= 2*(row_elements * participant_row_elements**.5)
            #similarity /= participant_autosimilarity

            df_combined_similarity_result = pd.concat([df_combined_similarity_result,
                                                       pd.DataFrame({'model': model, 'seq': seq,
                                                                     'similarity': similarity,
                                                                     'participant': participant_num},
                                                                    index=[0])], ignore_index=True)


    df_combined_similarity_mean = df_combined_similarity_result.groupby(['model', 'participant']).mean().reset_index()
    print(df_combined_similarity_mean)
    df_combined_similarity_mean_all = df_combined_similarity_mean.groupby(['model']).mean().reset_index()
    model_name = "Episodic Inference"
    final_mean_similarity = df_combined_similarity_mean_all[df_combined_similarity_mean_all['model'] == model_name][
        'similarity']
    return final_mean_similarity

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# Parameters to search.
param_grid = {'k': list(range(1, 4, 2)),
              'm': list(range(0, 4, 2)),
              'n': list(range(0, 4, 2)),
              'o': list(range(0, 4, 2))}

combination_of_parameters_to_try = ParameterGrid(param_grid)

results_storage = []
best_similarity_so_far = 0
best_params_so_far = {}
for params_to_try in tqdm(combination_of_parameters_to_try, desc='Searching the best parameters'):

    blockPrint()
    env = create_environment(**params_to_try)
    seqs_score = {"Episodic Inference": simulate_sequences(env, 4, 4),
                  "Episodic RL": episodic_rl_algorithm(df_env_states, df_env_novel_states, max_seqs=4,
                                                       n_samp=4, thresh=0.4)}
    df_sequences = pd.concat([create_df_sequences(seqs).assign(model=model) for model, seqs in seqs_score.items()],
                             ignore_index=True)
    enablePrint()
    final_mean_similarity = compute_evaluation(df_sequences, df_env_states_participants, df_words_similarities,
                                               df_locations_similarities, participant_numbers).values[0]
    print(final_mean_similarity)


    # Store the results of this combination
    d = {'params': params_to_try, 'result': final_mean_similarity}
    print(d)
    results_storage.append(d)
    if final_mean_similarity > best_similarity_so_far:
        best_params_so_far, best_similarity_so_far = params_to_try, final_mean_similarity
        print(f'best parameters so far: {best_params_so_far} with a similarity of {best_similarity_so_far}')


print(f'BEST PARAMETERS: {best_params_so_far} achieving SIMILARITY: {best_similarity_so_far}')
# Save results
json.dump(results_storage, open( "results_of_kmno_search.json", 'w'))



# TO DO:
# 1. Repassar la funció compute_evaluation. S'ha de normalitzar amb sentit.
#      1.1. (?) Normalitzar per lo llarga que sigui la seqüència? Sinó els participants amb moltes paraules donen molta més semblança amb el model
#      1.2. (?) si s'assembla molt a un participant i molt poc a un altre, donarà poca semblança per la mean. Està bé això?
