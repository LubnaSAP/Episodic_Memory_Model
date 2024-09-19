import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity




def processed_data(original_data):
    data = original_data.dropna(subset=['Sd15_Cosine'])
    data = data.drop_duplicates(subset=['word'])
    return data
def tokenize_locations(data):
    # Drop duplicates in the 'location' column
    unique_locations_data = data.drop_duplicates(subset=['location'])
    # Tokenize the locations
    tokenized_locations = [location.split() for location in unique_locations_data['location']]
    return tokenized_locations

# Function to compute location embeddings
def compute_location_embeddings(tokenized_locations):
    embedding_model = Word2Vec(sentences=tokenized_locations, vector_size=100, window=5, min_count=1, workers=4)
    location_embeddings = []
    for location in tokenized_locations:
        embedding_sum = np.zeros(embedding_model.vector_size)
        word_count = 0
        for word in location:
            if word in embedding_model.wv:
                embedding_sum += embedding_model.wv[word]
                word_count += 1
        if word_count > 0:
            location_embedding = embedding_sum / word_count
            location_embeddings.append(location_embedding)
    return location_embeddings

# Function to compute location similarity matrix
def compute_location_similarity(tokenized_locations, location_embeddings):
    location_similarity_matrix = cosine_similarity(location_embeddings)
    tokenized_locations_strings = [' '.join(location) for location in tokenized_locations]
    location_similarity_df = pd.DataFrame(location_similarity_matrix, index=tokenized_locations_strings, columns=tokenized_locations_strings)
    min_value = location_similarity_df.min().min()
    max_value = location_similarity_df.max().max()
    location_similarity_df = (location_similarity_df - min_value) / (max_value - min_value)
    return location_similarity_df

def compute_word_similarity(data):
    # Compute cosine similarity matrix for words
    word_matrix = cosine_similarity(data[['Sd15_Cosine', 'Sd15_CosRev0Score']])
    # Create DataFrame for cosine similarity matrix
    word_similarity_df = pd.DataFrame(word_matrix, index=data['word'], columns=data['word'])
    word_similarity_df.index.name = None  # Remove index name
    word_similarity_df.columns.name = None  # Remove column name
    return word_similarity_df


def compute_evaluation(df_sequences, df_env_novel_states, df_words_similarities, df_locations_similarities):
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


# Load data and compute similarity matrices for each participant
participant_numbers = [1, 2, 3, 4, 13]
for participant_num in participant_numbers:
    # Load the dataset
    original_data = pd.read_csv(f"participant_{participant_num}_pairs.csv")

    data = processed_data(original_data)

    # Tokenize the locations
    tokenized_locations = tokenize_locations(data)

    # Compute location embeddings
    location_embeddings = compute_location_embeddings(tokenized_locations)

    # Compute location similarity matrix
    location_similarity_df = compute_location_similarity(tokenized_locations, location_embeddings)

    # Save the location similarity matrix to a CSV file
    location_similarity_df.to_csv(f'locations_similarities_participant_{participant_num}.csv')

    word_similarity_df = compute_word_similarity(data)

    word_similarity_df.to_csv(f'words_similarities_participant_{participant_num}.csv')

    # Save the environment data to a CSV file
    data.to_csv(f'participant_{participant_num}_environment.csv')


