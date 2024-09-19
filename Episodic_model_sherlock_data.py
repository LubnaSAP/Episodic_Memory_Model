import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from Environment_Episodic import EpisodicGraph
from Generator_Episodic import Generator
from Propagator_Episodic import Propagator
from Simulator_Episodic import EpisodicSimulator
from episodic_replay_model import episodic_rl_algorithm
import seaborn.objects as so
import matplotlib.pyplot as plt


#definition of functions
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
        #label_i = df_env_states.loc[i_prev]["episode"]
        if i_prev in df_env_states.index:
            label_i = df_env_states.loc[i_prev]["episode"]

        label = f"Ep. {label_i}"
        ax.text(i_prev + i / 2, rectangle_height - 3, label, fontsize=9, ha='center', alpha = 0.5, color=c)

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

def compute_evaluation_model_participant(df_sequences, df_env_states_participants, df_words_similarities, df_locations_similarities, participant_numbers):
    df_combined_similarity_result = pd.DataFrame()  # DataFrame to store combined similarity scores

    for participant_num in participant_numbers:
        # Group by model and sequence
        df_sequences_by_model = df_sequences.groupby(['model', 'seq'])

        for (model, seq), df_seq_i in df_sequences_by_model:
            # Remove duplicated rows
            df_seq_i = df_seq_i.drop_duplicates(subset=['word', 'location'])

            similarity = 0  # Initialize similarity for the current sequence
            simn = 0

            # Iterate over the rows of the current sequence
            for _, row in df_seq_i.iterrows():
                # Iterate over the rows of the participant data
                for _, participant_row in df_env_states_participants[participant_num].iterrows():
                    # Get the semantic and spatial similarity between the model's word/location and participant's word/location
                    semantic_sim = df_words_similarities.loc[row['word'], participant_row['word']]
                    spatial_sim = df_locations_similarities.loc[row['location'], participant_row['location']]
                    # Compute the similarity
                    similarity += semantic_sim + spatial_sim
                    simn += 2

            similarity = similarity/simn
            # Append data to the DataFrame
            df_combined_similarity_result = pd.concat([df_combined_similarity_result, pd.DataFrame({'model': model, 'seq': seq, 'similarity': similarity, 'participant': participant_num}, index=[0])], ignore_index=True)

    # Compute mean similarity separately for each model
    df_combined_similarity_mean = df_combined_similarity_result.groupby(['model', 'participant']).mean().reset_index()
    df_combined_similarity_mean_all = df_combined_similarity_mean.groupby(['model']).mean().reset_index()

    model_name = "Episodic Inference"

    final_mean_similarity_episodic_inference = df_combined_similarity_mean_all[df_combined_similarity_mean_all['model'] == model_name]
    return df_combined_similarity_result, df_combined_similarity_mean, df_combined_similarity_mean_all, final_mean_similarity_episodic_inference['similarity']




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

    # Add the filtered DataFrame to the dictionary
    df_env_states_participants[participant_num] = df_participant

# Concatenate all DataFrames to get a single DataFrame
df_all_participants = pd.concat(df_env_states_participants.values())

# Count the number of unique words with salience = 1
total_unique_words = len(df_all_participants['word'].unique())

# Print the total unique number of words with salience = 1
print(f"Total unique number of words with salience = 1: {total_unique_words}")



#print(df_env_states_participants[2])

######################################################################################################################################


import itertools


# Create the environment
k = 1.0  # Inter-episode connectivity. Domain: [0, 1]. 1, sample across episodes. 0, sample within episodes
m = 0.0  # Semantic Similarity weight. Domain: [0, inf]. 0, no action dependence.
n = 0.0  # Temporal Similarity weight. Domain: [0, inf]. 0, no time dependence.
o = 0.0  # Spatial Similarity weight. Domain: [0, inf]. 0, no spatial dependence.

models = {
    "Episodic Inference": (k, m, n, o),
}


n_step = 40
n_samp = 100
seqs_score = {}
for model, params in models.items():
    k , m, n, o = params
    env = EpisodicGraph(df_env_states, df_words_similarities_tuned, df_locations_similarities, k=k, m=m, n=n, o=o)
    # env = RoomWorld()

    # Create the generator
    generator = Generator(env)

    # Create the propagator
    propagator = Propagator(generator)
    # propagator.min_zero_cf()

    # Create the simulator

    # random init state
    init_state = np.random.randint(0, len(env.states))
    simulator = EpisodicSimulator(propagator, init_state)

    # Simulate
    simulator.sample_sequences(n_step=n_step, n_samp=n_samp)

    seqs_score[model] = simulator.state_seqs


episodic_rl_label = "Episodic RL"
rl_thresh = 0.4
seqs_score[episodic_rl_label] = episodic_rl_algorithm(df_env_states, df_env_novel_states, max_seqs=n_step,
                                                      n_samp=n_samp, thresh=rl_thresh)





# Create a pandas dataframe with the states sequence

df_sequences = pd.DataFrame()
for model, seqs in seqs_score.items():
    df_sequences_i = create_df_sequences(seqs)
    df_sequences_i["model"] = model

    #df_sequences = df_sequences.append(df_sequences_i, ignore_index=True)
    # Instead of df_sequences.append(df_sequences_i, ignore_index=True)
    df_sequences = pd.concat([df_sequences, df_sequences_i], ignore_index=True)


fig, ax = plt.subplots(figsize=(10, 5))

# Extract the unique words from the sampled df
words_sorted = df_env_states['word'].unique()
# Order it alphabetically
words_sorted.sort()
# Create a dict with the index of each word
words_sorted = {word: i for i, word in enumerate(words_sorted)}

plot_environment(df_env_states, words_sorted, ax=ax)

# plot the df_sequences

# extrat model "Episodic Inference" and "Episodic RL"
df_sequences_2 = df_sequences[df_sequences['model'].isin(["Episodic Inference", episodic_rl_label])].groupby('model')
for c, (name, df_sequences_i) in zip(["b", "r"], df_sequences_2):
    # reset index
    df_sequences_i = df_sequences_i.reset_index(drop=True)
    df_sequences_i = df_sequences_i.groupby('seq')

    # Extract fir
    for seq_i, df_sequences_seq_i in df_sequences_i:
        for i, row in df_sequences_seq_i.iterrows():
            ax.scatter(row['time'] * (len(df_env_states) - 1), words_sorted[row['word']], color=c, s=4, alpha=1)
            # add lines conecting the points
            if i > 0:
                if i == 1:
                    ax.plot([df_sequences_seq_i.iloc[i - 1]['time'] * (len(df_env_states) - 1),
                             row['time'] * (len(df_env_states) - 1)],
                            [words_sorted[df_sequences_seq_i.iloc[i - 1]['word']], words_sorted[row['word']]], color=c,
                            alpha=0.3, label=name)
                else:

                    ax.plot([df_sequences_seq_i.iloc[i - 1]['time'] * (len(df_env_states) - 1),
                             row['time'] * (len(df_env_states) - 1)],
                            [words_sorted[df_sequences_seq_i.iloc[i - 1]['word']], words_sorted[row['word']]], color=c,
                            alpha=0.3)
        break

ax.legend()
# set x label
ax.set_xlabel("Time (s)")
plt.show()
# save as pdf
fig.savefig("sampling.pdf", bbox_inches="tight")


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
fig.savefig("evaluation.pdf", bbox_inches="tight")


##########################################################################################################################################################



# Call the function to compute similarity scores
df_combined_similarity, df_combined_similarity_mean, df_combined_similarity_mean_all, final_mean_similarity_episodic_inference = compute_evaluation_model_participant(df_sequences, df_env_states_participants, df_words_similarities, df_locations_similarities, participant_numbers)

# Print the combined similarity DataFrame
print("\nMean Similarity all participants:")
print(df_combined_similarity_mean_all)
print("\nFinal Mean Similarity Episodic Inference:")
print(final_mean_similarity_episodic_inference)



