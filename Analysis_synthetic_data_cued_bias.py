import pandas as pd
import numpy as np
import seaborn as sns
from Environment_Episodic_cued_biased import EpisodicGraph_cued_biasy
from Environment_Episodic import EpisodicGraph
from Generator_Episodic import Generator
from Propagator_Episodic import Propagator
from Simulator_Episodic import EpisodicSimulator
import matplotlib.pyplot as plt



df_words_similarities = pd.read_csv('words_similarities_semanthic.csv', index_col=0)
df_locations_similarities = pd.read_csv('locations_similarities_semanthic.csv', index_col=0)
df_env_states = pd.read_csv('synthetic_sanity_check.csv', index_col=0)
df_env_novel_states = pd.read_csv('synthetic_semantic_cue.csv', index_col=0)



# Create the environment
k = 0.2  # Inter-episode connectivity. Domain: [0, 1]. 1, sample across episodes. 0, sample within episodes
m = 0  # Semantic Similarity weight. Domain: [0, inf]. 0, no action dependence.
n = 0  # Temporal Similarity weight. Domain: [0, inf]. 0, no time dependence.
o = 0  # Spatial Similarity weight. Domain: [0, inf]. 0, no spatial dependence.

models = {
    "Episodic Inference": (0, 0, 0),
    "Episodic Inference\nsemantic-biased": (1, 0, 0),
    "Episodic Inference\ntemporal-biased": (0, 1, 0),
    "Episodic Inference\nspatial-biased": (0, 0, 1),
}
n_step = 3
n_samp = 300
seqs_score = {}

# With novel states
for model, params in models.items():
    k = 0
    m, n, o = params
    p=5
    env_novel = EpisodicGraph_cued_biasy(df_env_states, df_words_similarities, df_locations_similarities,df_env_novel_states, k=k, m=m, n=n, o=o, p=p)
    #env_novel.plot_stochastic_matrix()

    generator = Generator(env_novel)
    propagator = Propagator(generator)
    init_state = 4
    simulator = EpisodicSimulator(propagator, init_state)
    simulator.sample_sequences(n_step=n_step, n_samp=n_samp)
    seqs_score[f"{model}_with_novel"] = simulator.state_seqs





# Without novel states
for model, params in models.items():
    k = 0.2
    m, n, o = params
    env = EpisodicGraph(df_env_states, df_words_similarities, df_locations_similarities, k=k, m=m, n=n, o=o)
    #env.plot_stochastic_matrix()
    generator = Generator(env)
    propagator = Propagator(generator)
    init_state = 4
    simulator = EpisodicSimulator(propagator, init_state)
    simulator.sample_sequences(n_step=n_step, n_samp=n_samp)
    seqs_score[f"{model}_without_novel"] = simulator.state_seqs



def create_df_sequences(seqs, df_env_states):
    df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
    for i, seq in enumerate(seqs):  # Iterate over each sequence
        df_i = df_env_states.iloc[seq]  # Select rows corresponding to states in the sequence
        df_i["seq"] = i  # Append a column with the sequence number
        df_sequences = pd.concat([df_sequences, df_i], ignore_index=True)  # Append the sequence to the DataFrame

        # Debugging output
        print(f"Trajectory {i}: Length = {len(df_i)}, States: {df_i['word'].tolist()}")

    return df_sequences


df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
for model, seqs in seqs_score.items():
    df_sequences_i = create_df_sequences(seqs, df_env_states)
    df_sequences_i["model"] = model  # Add a column for the model name
    df_sequences = pd.concat([df_sequences, df_sequences_i], ignore_index=True)  # Append the sequences to the DataFrame

# Compute the evaluation including standard deviation
def compute_evaluation(df_sequences, df_env_novel_states):
    df_sequences_by_model = df_sequences.groupby(['model', 'seq'])
    df_final_data = []

    for seq_i, df_seq_i in df_sequences_by_model:
        reward = 0
        len_sf_seq_i = len(df_seq_i)
        for i, row in df_seq_i.iterrows():
            for j, novel_row in df_env_novel_states.iterrows():
                semantic_sim = df_words_similarities.loc[row['word'], novel_row['word']]
                spatial_sim = df_locations_similarities.loc[row['location'], novel_row['location']]
                temporal_sim = 1 - abs(row["time"] - novel_row["time"]) / max(row["time"], novel_row["time"])
                episodic_sim = 1 - abs(row["episode"] - novel_row["episode"]) / (max(row["episode"], novel_row["episode"] + 1e-5))
                similarity = (semantic_sim + spatial_sim + temporal_sim + episodic_sim) / 4

                reward += similarity
        final_reward = reward / len_sf_seq_i
        df_final_data.append({'model': df_seq_i['model'].values[0], 'seq': seq_i, 'reward': final_reward})

    df_final = pd.DataFrame(df_final_data)

    # Group by model and compute both the mean reward and standard deviation
    df_sequences_by_model_sim = df_final.groupby('model').agg(
        reward_mean=('reward', 'mean'),
        reward_std=('reward', 'std')  # Calculate standard deviation
    ).reset_index()


    return df_sequences_by_model_sim



# Compute the evaluation including standard deviation
df_eval = compute_evaluation(df_sequences, df_env_novel_states)

# Create a new column for the base model name
df_eval['base_model'] = df_eval['model'].str.replace('_with_novel', '').str.replace('_without_novel', '')

# Create a new column to indicate with_novel or without_novel
df_eval['novelty'] = df_eval['model'].apply(lambda x: 'with_novel' if 'with_novel' in x else 'without_novel')

# Define some parameters for the plot
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.6  # Width of the bars
n_hues = len(df_eval['novelty'].unique())  # Number of unique novelty conditions
bar_width_per_hue = bar_width / n_hues  # Width for each individual bar within the base model group

# Plot bars
sns.barplot(
    data=df_eval,
    x="base_model",
    y="reward_mean",
    hue="novelty",
    ax=ax,
    dodge=True,
    width=bar_width,
    palette="viridis"
)

# Get the positions of the bars (x-ticks)
xticks = ax.get_xticks()

# Manually add error bars
for i, base_model in enumerate(df_eval['base_model'].unique()):
    model_data = df_eval[df_eval['base_model'] == base_model]
    for j, novelty in enumerate(model_data['novelty'].unique()):
        subset = model_data[model_data['novelty'] == novelty]

        # Calculate the x position based on the "dodge" effect
        x_pos = xticks[i] - bar_width / 2 + bar_width_per_hue / 2 + j * bar_width_per_hue

        reward_mean = subset['reward_mean'].values[0]
        reward_std = subset['reward_std'].values[0]

        # Plot the error bar at the correct position
        ax.errorbar(
            x=x_pos,
            y=reward_mean,
            yerr=reward_std,
            fmt='o',
            color='black',
            capsize=5,
            label='_nolegend_'  # To avoid duplicate labels in the legend
        )

ax.set_title("Comparison of Reward with and without Novel States Similarity")
ax.set_xlabel("Model")
ax.set_ylabel("Similarity Score")

# Move the legend outside the plot
ax.legend(title='Novelty', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#fig.savefig("Bar_comparison_with_and_without_cued_biasy.pdf", bbox_inches="tight")
