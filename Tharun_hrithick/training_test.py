import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
import re
import time
# --- IMPROVEMENT: Added Learning Rate Scheduler ---
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 1. Configuration & Hyperparameters ---
# --- CRITICAL: This file MUST exist. It's not test.csv ---
FILE_PATH = 'preprocessed_test.csv' 
MODEL_SAVE_PATH = 'emotion_detection_pytorch_improved.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
VOCAB_MIN_FREQ = 3
MAX_LEN = 60
# --- IMPROVEMENT: Increased model capacity ---
EMBEDDING_DIM = 200 # Original: 100
HIDDEN_DIM = 256    # Original: 128
NUM_CLASSES = 6     # Based on your dataset (0-5)
NUM_LAYERS = 2
DROPOUT_PROB = 0.5

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
# --- IMPROVEMENT: Increased epochs, but added Early Stopping ---
NUM_EPOCHS = 50 # Original: 30 (Early stopping will likely stop it sooner)
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15 # Percentage of the *training* data

# --- IMPROVEMENT: New parameters for better training ---
WEIGHT_DECAY = 1e-5             # L2 Regularization to prevent overfitting
GRADIENT_CLIP_VALUE = 1.0       # Prevents exploding gradients for stable training
EARLY_STOPPING_PATIENCE = 5   # Stop if val_loss doesn't improve for 5 epochs

# --- 2. Data Loading and Inspection ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset loaded successfully from '{FILE_PATH}'.")
    print(f"Total samples: {len(df)}\n")
    
    # Ensure column names are correct (text, label)
    df = df.rename(columns={'comment': 'text', 'emotion': 'label'}, errors='ignore')
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    
    # Basic class distribution
    print("Class distribution:")
    label_counts = df['label'].value_counts().sort_index()
    print(label_counts)
    print("\n")

# --- 3. Preprocessing and Vocabulary ---
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)

    def build_vocab(texts, min_freq):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        
        words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # Add special tokens
        # <pad> is 0, <unk> is 1
        vocab = {'<pad>': 0, '<unk>': 1}
        for i, word in enumerate(words):
            vocab[word] = i + 2
        return vocab

    print("Building vocabulary...")
    VOCAB = build_vocab(df['text'], VOCAB_MIN_FREQ)
    VOCAB_SIZE = len(VOCAB)
    print(f"Vocabulary size: {VOCAB_SIZE}\n")

# --- 4. Dataset Class ---
    class TextDataset(Dataset):
        def __init__(self, texts, labels, vocab, max_len):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            # Convert text to sequence
            seq = [self.vocab.get(word, 1) for word in text.split()] # 1 for <unk>
            
            # Pad sequence
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            else:
                seq = seq + [0] * (self.max_len - len(seq)) # 0 for <pad>
            
            return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    all_texts = df['text'].values
    all_labels = df['label'].values
    
    dataset = TextDataset(all_texts, all_labels, VOCAB, MAX_LEN)

# --- 5. Split Data and Create DataLoaders ---
    test_size = int(len(dataset) * TEST_SPLIT)
    train_val_size = len(dataset) - test_size
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

    val_size = int(train_val_size * VAL_SPLIT)
    train_size = train_val_size - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# --- 6. Model Definition ---
    class EmotionModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout_prob):
            super(EmotionModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_prob if num_layers > 1 else 0 # Dropout between LSTM layers
            )
            # Final dropout layer
            self.dropout = nn.Dropout(dropout_prob)
            self.fc = nn.Linear(hidden_dim * 2, num_classes) # *2 for bidirectional

        def forward(self, x):
            embedded = self.embedding(x)
            # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]
            # hidden shape: [num_layers * 2, batch_size, hidden_dim]
            lstm_out, (hidden, cell) = self.lstm(embedded)
            
            # Concatenate the final forward and backward hidden states
            # Get the last layer's hidden state (forward and backward)
            # hidden[-2,:,:] is the last forward layer
            # hidden[-1,:,:] is the last backward layer
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            
            # Apply dropout
            dropped = self.dropout(hidden)
            output = self.fc(dropped)
            return output

    model = EmotionModel(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_CLASSES,
        NUM_LAYERS,
        DROPOUT_PROB
    ).to(DEVICE)
    
    print("Model architecture created:")
    print(model)
    print("\n")

# --- 7. Training Setup ---
    criterion = nn.CrossEntropyLoss()
    # --- IMPROVEMENT: Added weight_decay (L2 regularization) ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # --- IMPROVEMENT: Added Learning Rate Scheduler ---
    # Will reduce LR if validation loss plateaus for 2 epochs
    # *** THIS IS THE FIX: Removed 'verbose=True' which caused the error ***
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# --- 8. Training Loop ---
    print("--- Starting Model Training ---")
    start_time = time.time()
    
    best_val_loss = float('inf')
    # --- IMPROVEMENT: Added for Early Stopping ---
    epochs_no_improve = 0 

    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        total_train_loss = 0
        
        for texts_batch, labels_batch in train_loader:
            texts_batch = texts_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            # Forward pass
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # --- IMPROVEMENT: Added Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        total_val_correct = 0
        
        with torch.no_grad():
            for texts_batch, labels_batch in val_loader:
                texts_batch = texts_batch.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                
                outputs = model(texts_batch)
                loss = criterion(outputs, labels_batch)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val_correct += (predicted == labels_batch).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * total_val_correct / len(val_dataset)
        
        print(f'Epoch [{epoch+1:02}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')
        
        # --- IMPROVEMENT: Learning Rate Scheduler Step ---
        scheduler.step(avg_val_loss)
        
        # --- IMPROVEMENT: Early Stopping and Model Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            # Save the best model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab': VOCAB,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_classes': NUM_CLASSES,
                'num_layers': NUM_LAYERS,
                'dropout_prob': DROPOUT_PROB,
                'max_len': MAX_LEN
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"   -> Validation loss improved. Saving new best model to {MODEL_SAVE_PATH}")
        
        else:
            epochs_no_improve += 1
            print(f"   -> Validation loss did not improve. Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n--- Early stopping triggered after {epoch+1} epochs. ---")
            break # Exit the training loop
    
    training_time = (time.time() - start_time) / 60
    print(f"--- Training Finished in {training_time:.2f} minutes ---")

# --- 9. Test Evaluation ---
    print(f"\nLoading best model from {MODEL_SAVE_PATH} for final testing...")
    
    # Load the saved checkpoint
    # We use weights_only=False to load the full checkpoint dictionary (with vocab)
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    
    # Re-create model structure and load weights
    test_model = EmotionModel(
        vocab_size=len(checkpoint['vocab']), # Use vocab size from checkpoint
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes'],
        num_layers=checkpoint['num_layers'],
        dropout_prob=checkpoint['dropout_prob']
    ).to(DEVICE)
    
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval() # Set to evaluation mode

    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for texts_batch, labels_batch in test_loader:
            texts_batch = texts_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            outputs = test_model(texts_batch)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels_batch.size(0)
            test_correct += (predicted == labels_batch).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f'\nAccuracy of the best model on the test dataset: {test_accuracy:.2f} %')
    print("--- Process Completed ---")

except FileNotFoundError:
    print(f"Error: Could not find the dataset file at '{FILE_PATH}'.")
    print("Please make sure 'preprocessed_test.csv' is in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")