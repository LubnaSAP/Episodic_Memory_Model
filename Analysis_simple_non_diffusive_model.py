from Environment_Episodic_cued_biased import EpisodicGraph_cued_biasy
from Environment_Episodic import EpisodicGraph
from Generator_Episodic import Generator
from Propagator_Episodic import Propagator
from Simulator_Episodic import EpisodicSimulator
from episodic_replay_model import episodic_rl_algorithm
from Simple_uniform_non_diffusive_model import SimpleGenerator, SimpleSimulator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn.objects as so

def compute_evaluation(df_sequences, df_env_novel_states):
    df_sequences_by_model = df_sequences.groupby(['model', 'seq'])
    df_final_data = []  # List to accumulate data

    # Iterate over the groups
    for seq_i, df_seq_i in df_sequences_by_model:
        reward = 0
        #df_seq_i = df_seq_i.drop_duplicates(subset=['word', 'location'])
        for i, row in df_seq_i.iterrows():
            for j, novel_row in df_env_novel_states.iterrows():
                semantic_sim = df_words_similarities.loc[row['word'], novel_row['word']]
                spatial_sim = df_locations_similarities.loc[row['location'], novel_row['location']]
                temporal_sim = 1 - abs(row["time"] - novel_row["time"])/(max(row["time"],novel_row["time"]+1e-5))
                episodic_sim = 1 - abs(row["episode"] - novel_row["episode"]) / (max(row["episode"], novel_row["episode"] + 1e-5))
                similarity = (semantic_sim + spatial_sim + temporal_sim + episodic_sim) / 4

                reward += similarity
            reward = reward /len(df_seq_i)
        df_final_data.append({'model': df_seq_i['model'].values[0], 'seq': seq_i, 'reward': reward})

    df_final = pd.DataFrame(df_final_data)
    df_sequences_by_model_sim = df_final.groupby(['model', 'seq']).mean().reset_index()

    return df_sequences_by_model_sim
def create_df_sequences(seqs, df_env_states):
    df_sequences = pd.DataFrame()
    for i, seq in enumerate(seqs):
        df_i = df_env_states.iloc[seq]
        df_i["seq"] = i
        df_sequences = df_sequences._append(df_i, ignore_index=True)
        print(f"Trajectory {i}: Length = {len(df_i)}, States: {df_i['word'].tolist()}")
    return df_sequences

df_words_similarities = pd.read_csv('words_similarities_semanthic.csv', index_col=0)
df_locations_similarities = pd.read_csv('locations_similarities_semanthic.csv', index_col=0)
df_env_states = pd.read_csv('synthetic_sanity_check.csv', index_col=0)
df_env_novel_states = pd.read_csv('synthetic_semantic_cue.csv', index_col=0)

models = {
    #"Episodic Inference": (1.0, 0, 0, 0),
    "Episodic Inference\nsemantic-biased_1.0": (0.1, 1.0, 0, 0),
    #"Episodic Inference\ntemporal-biased": (0.1, 0, 1, 0),
    #"Episodic Inference\nspatial-biased": (0.1, 0, 0, 1),
    "Simple Model": (0.1, 1, 0, 0),  # Parameters for the simple model
    "Episodic RL": (0, 0, 0, 0),  # Parameters for the Episodic RL model
}

models_fine_tuned = {
    "Episodic Inference": (1.0, 0, 0, 0, 1.0),
    #"Episodic Inference\nsemantic-biased_1.0": (0.1, 1.0, 0, 0, 1.0),
    #"Episodic Inference\ntemporal-biased": (0.1, 0, 1, 0, 1.0),
    #"Episodic Inference\nspatial-biased": (0.1, 0, 0, 1, 1.0),
    "Simple Model": (0.1, 1, 0, 0, 1.0),  # Parameters for the simple model
    #"Episodic RL": (0, 0, 0, 0, 0),  # Parameters for the Episodic RL model
}

# Define parameters
n_step = 3
n_samp = 150
init_state = 4
seqs_score = {}

# Existing models and the new fine-tuned model
for model, params in models.items():
    if model == "Fine Tuned Model":
        k, m, n, o, p = params
        env = EpisodicGraph_cued_biasy(df_env_states, df_words_similarities,
                                       df_locations_similarities, df_env_novel_states, k=k, m=m, n=n,
                                       o=o, p=p)
    else:
        k, m, n, o = params
        env = EpisodicGraph(df_env_states, df_words_similarities, df_locations_similarities, k=k, m=m, n=n, o=o)

    # Model-specific processing
    # Model-specific processing
    if model == "Simple Model":
        # init_state = 0  # Default init_state for Simple Model
        generator = SimpleGenerator(env)  # Enable extra state for Simple Model
        simulator = SimpleSimulator(generator, init_state)
        simulator.sample_sequences(n_step=n_step, n_samp=n_samp)
        # Plot the stochastic matrix for the Simple Model
        #env.plot_stochastic_matrix()

    elif model == "Fine Tuned Model":
        # init_state = 5  # Default init_state for Simple Model Fine Tuned
        generator = Generator(env)
        env.plot_stochastic_matrix()
        propagator = Propagator(generator)
        simulator = EpisodicSimulator(propagator, init_state=init_state)

    elif model == "Episodic RL":
        seqs = episodic_rl_algorithm(df_env_states, df_env_novel_states, df_words_similarities, max_seqs=n_step, n_samp=n_samp, thresh=0.4)
        seqs_score[model] = seqs
        continue  # Skip the rest of the loop for Episodic RL

    else:
        generator = Generator(env)
        #env.plot_stochastic_matrix()
        propagator = Propagator(generator)
        simulator = EpisodicSimulator(propagator, init_state=init_state)

    simulator.sample_sequences(n_step=n_step, n_samp=n_samp)
    seqs_score[model] = simulator.state_seqs

# Evaluation and Visualization
# Create a pandas DataFrame with the states sequence
df_sequences = pd.DataFrame()  # Initialize an empty DataFrame
for model, seqs in seqs_score.items():
    if model == "Episodic RL":
        # Convert Episodic RL sequences to DataFrame
        df_sequences_i = create_df_sequences(seqs, df_env_states)
    else:
        df_sequences_i = create_df_sequences(seqs, df_env_states)
    df_sequences_i["model"] = model  # Add a column for the model name
    df_sequences = pd.concat([df_sequences, df_sequences_i], ignore_index=True)

episodic_rl_label = "Episodic RL"
rl_thresh = 0.4
seqs_score[episodic_rl_label] = episodic_rl_algorithm(df_env_states, df_env_novel_states, df_words_similarities, max_seqs=n_step,
                                                      n_samp=n_samp, thresh=rl_thresh)


# Create DataFrame with sequences for all models
df_sequences = pd.DataFrame()
for model, seqs in seqs_score.items():
    df_sequences_i = create_df_sequences(seqs, df_env_states)
    df_sequences_i["model"] = model
    df_sequences = df_sequences._append(df_sequences_i, ignore_index=True)


df_eval = compute_evaluation(df_sequences, df_env_novel_states)

# Visualization
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

# Save the plot as a PDF
#fig.savefig("evaluation_sanity_check_non_difusion.pdf", bbox_inches="tight")


#######################################################################
def compute_evaluation(df_sequences, df_env_novel_states):
    df_sequences_by_model = df_sequences.groupby(['model', 'seq'])
    df_final_data = []  # List to accumulate data

    # Iterate over the groups
    for seq_i, df_seq_i in df_sequences_by_model:
        reward = 0
        semantic_rewards = []
        spatial_rewards = []
        temporal_rewards = []

        df_seq_i = df_seq_i.drop_duplicates(subset=['word', 'location'])
        for i, row in df_seq_i.iterrows():
            for j, novel_row in df_env_novel_states.iterrows():
                semantic_sim = df_words_similarities.loc[row['word'], novel_row['word']]
                spatial_sim = df_locations_similarities.loc[row['location'], novel_row['location']]
                temporal_sim = 1 - abs(row["time"] - novel_row["time"]) / (max(row["time"], novel_row["time"]) + 1e-5)
                similarity = (semantic_sim + spatial_sim + temporal_sim) / 3
                reward += similarity
                semantic_rewards.append(semantic_sim)
                spatial_rewards.append(spatial_sim)
                temporal_rewards.append(temporal_sim)

        #final_reward = reward / len(df_seq_i)
        df_final_data.append({
            'model': df_seq_i['model'].values[0],
            'seq': seq_i,
            #'reward': final_reward,
            'reward': reward,
            'semantic_reward': np.mean(semantic_rewards),
            'spatial_reward': np.mean(spatial_rewards),
            'temporal_reward': np.mean(temporal_rewards)
        })

    df_final = pd.DataFrame(df_final_data)
    df_sequences_by_model_sim = df_final.groupby(['model', 'seq']).mean().reset_index()
    return df_sequences_by_model_sim


# Assuming df_sequences and df_env_novel_states are already defined
df_eval = compute_evaluation(df_sequences, df_env_novel_states)

# Compute mean and SEM for each model
df_summary = df_eval.groupby('model').agg(
    mean_reward=('reward', 'mean'),
    sem_reward=('reward', 'sem'),
    mean_semantic_reward=('semantic_reward', 'mean'),
    mean_spatial_reward=('spatial_reward', 'mean'),
    mean_temporal_reward=('temporal_reward', 'mean')
).reset_index()


# Visualization
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Bar plot with error bars for overall rewards
ax[0].bar(df_summary['model'], df_summary['mean_reward'], yerr=df_summary['sem_reward'], capsize=5, color='skyblue',
          edgecolor='black')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Reward')
ax[0].set_title('Model Evaluation with Uncertainty')
ax[0].set_xticklabels(df_summary['model'], rotation=45)

# Detailed contributions of each similarity component
width = 0.2  # Width of the bars
x = np.arange(len(df_summary['model']))  # Label locations

ax[1].bar(x - width, df_summary['mean_semantic_reward'], width, label='Semantic Reward', color='blue')
ax[1].bar(x, df_summary['mean_spatial_reward'], width, label='Spatial Reward', color='green')
ax[1].bar(x + width, df_summary['mean_temporal_reward'], width, label='Temporal Reward', color='red')

ax[1].set_xlabel('Model')
ax[1].set_ylabel('Component Reward')
ax[1].set_title('Component Rewards by Model')
ax[1].set_xticks(x)
ax[1].set_xticklabels(df_summary['model'], rotation=45)
ax[1].legend()

plt.tight_layout()

# Save the plot as a PDF
#fig.savefig("comparison_non_diffusive_uniform_model.pdf", bbox_inches="tight")
plt.show()
