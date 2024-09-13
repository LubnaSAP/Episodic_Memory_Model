import numpy as np

def episodic_rl_algorithm(df_env_states, df_env_novel_states, df_words_similarities, max_seqs, n_samp, thresh=0.4):
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

            random_group_id += 1  # Move to the next episode ID

        # Ensure seqs_i is exactly max_seqs length
        if len(seqs_i) < max_seqs:
            seqs_i += [-1] * (max_seqs - len(seqs_i))
        else:
            seqs_i = seqs_i[:max_seqs]

        seqs[i] = np.array(seqs_i)

    return seqs
