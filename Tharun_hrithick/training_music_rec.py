import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configuration ---
MUSIC_DATA_PATH = 'cleaned_music_sentiment_dataset.csv'

# Output files for the backend
OUTPUT_CSV_PATH = 'new_dataset_for_rec.csv'
OUTPUT_SIMILARITY_MATRIX_PATH = 'new_dataset_similarity_matrix_tags.npy'
OUTPUT_INDICES_PATH = 'new_dataset_song_indices_tags.pkl'
OUTPUT_PREPROCESSOR_PATH = 'tag_preprocessor.pkl' # To process tags

# --- 2. Main Execution Block ---
if __name__ == '__main__':
    print("\n--- Phase 1: Loading and Preparing Music Data ---")
    df_music = pd.read_csv(MUSIC_DATA_PATH)
    
    # Define all "tags" as per the PDF
    tag_cols = ['Sentiment_Label', 'Genre', 'Mood', 'Energy', 'Danceability']
    numerical_cols = ['Tempo (BPM)']
    key_cols = ['Song_Name', 'Artist'] + tag_cols + numerical_cols
    
    # Drop rows where any key data is missing
    df_music.dropna(subset=key_cols, inplace=True)
    
    # Ensure there are no duplicate songs
    df_music.drop_duplicates(subset=['Song_Name', 'Artist'], inplace=True, keep='first')
    df_music.reset_index(drop=True, inplace=True)

    print(f"Loaded and filtered dataset with {len(df_music)} unique songs.")
    
    # Save this clean, unique list of songs for the backend
    df_music.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved clean song list to '{OUTPUT_CSV_PATH}'")

    # --- Phase 2: Build Feature Matrix from TAGS ---
    print("\n--- Phase 2: Building Feature Matrix from TAGS ---")
    
    # Define categorical and numerical features
    categorical_features = ['Sentiment_Label', 'Genre', 'Mood', 'Energy', 'Danceability']
    numerical_features = ['Tempo (BPM)']

    # Create a pre-processing pipeline
    # 1. Numerical: Scale 'Tempo (BPM)' to be between 0 and 1
    numerical_transformer = MinMaxScaler()
    
    # 2. Categorical: Convert all text tags into binary (0/1) columns
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Fit and transform the data
    print("Fitting preprocessor and transforming tags into a feature matrix...")
    feature_matrix = preprocessor.fit_transform(df_music)
    
    print(f"Created feature matrix with shape: {feature_matrix.shape}")

    # --- Phase 3: Calculate and Save Similarity Matrix ---
    print("\n--- Phase 3: Calculating and Saving Similarity Matrix ---")
    print("Calculating cosine similarity matrix based on content features (TAGS)...")
    
    # We use .toarray() if the matrix is sparse, which OneHotEncoder produces
    if hasattr(feature_matrix, "toarray"):
        feature_matrix_dense = feature_matrix.toarray()
    else:
        feature_matrix_dense = feature_matrix
        
    cosine_sim = cosine_similarity(feature_matrix_dense, feature_matrix_dense)
    
    # 6. Save the similarity matrix
    np.save(OUTPUT_SIMILARITY_MATRIX_PATH, cosine_sim)
    print(f"Saved new similarity matrix to '{OUTPUT_SIMILARITY_MATRIX_PATH}'")
    
    # 7. Create and save the series for song title-to-index mapping
    indices = pd.Series(df_music.index, index=df_music['Song_Name']).drop_duplicates()
    with open(OUTPUT_INDICES_PATH, 'wb') as f:
        pickle.dump(indices, f)
    print(f"Saved song indices to '{OUTPUT_INDICES_PATH}'")
    
    # 8. Save the preprocessor itself for later use (if needed)
    with open(OUTPUT_PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved tag preprocessor to '{OUTPUT_PREPROCESSOR_PATH}'")
    
    print("\n--- Pre-computation based on TAGS is complete! ---")

