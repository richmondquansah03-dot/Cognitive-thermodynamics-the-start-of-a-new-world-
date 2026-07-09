import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# =========================================================================
# V154: DUAL-DATASET CT-TRANSFORMER TRAINING
# Merging Hollywood (High Drama) and DailyDialog (Organic Baseline)
# =========================================================================

# The exact files you provided
HOLLYWOOD_FILE = r"C:\Users\GIGABYTE\Documents\ct\Ouroboros_Outputs\AEI_Generative\V145_CINEMATIC_MASTER_NBODY_20260629_182756.csv"
DAILYDIALOG_FILE = r"C:\Users\GIGABYTE\Documents\ct\Ouroboros_Outputs\AEI_Generative\V153_DAILYDIALOG_MASTER_VECTORS_20260704_211422.csv"
OUTPUT_DIR = r"C:\Users\GIGABYTE\Documents\ct\Ouroboros_Outputs\AEI_Generative"

# --- TRANSFORMER HYPERPARAMETERS ---
MAX_SEQ_LENGTH = 16   
D_MODEL = 256         
N_HEAD = 8            
NUM_LAYERS = 3        
DIM_FEEDFORWARD = 512 
BATCH_SIZE = 128      # Increased for the larger combined dataset      
EPOCHS = 15           
LEARNING_RATE = 0.0005 

# --- V91 3D COGNITIVE TOPOLOGY ---
OBSERVER_LABELS = ['curiosity', 'realization', 'confusion', 'admiration', 'excitement', 'optimism', 'gratitude', 'remorse', 'desire']
ADMIN_LABELS = ['anger', 'annoyance', 'disapproval', 'disgust', 'fear', 'nervousness', 'approval', 'caring', 'pride', 'joy', 'love', 'sadness', 'relief']
EXTERNAL_CHANGE_LABELS = ['disappointment', 'embarrassment', 'grief']
NEUTRAL_LABEL = ['neutral']
MODIFIER_LABELS = ['amusement', 'surprise']

ALL_EMOTIONS = OBSERVER_LABELS + ADMIN_LABELS + EXTERNAL_CHANGE_LABELS + NEUTRAL_LABEL + MODIFIER_LABELS

# =========================================================================
# 1. POSITIONAL ENCODING & TRANSFORMER ARCHITECTURE
# =========================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CinematicDirectorTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        # Project pure continuous physics to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_SEQ_LENGTH)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
        
    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x_cont, seq_lengths):
        batch_size, max_len, _ = x_cont.shape
        
        # Dynamic Right-Padding Mask
        arange_tensor = torch.arange(max_len, device=x_cont.device).unsqueeze(0).expand(batch_size, max_len)
        lengths_tensor = seq_lengths.unsqueeze(1).expand(batch_size, max_len)
        padding_mask = arange_tensor >= lengths_tensor
        
        x = self.input_projection(x_cont) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        causal_mask = self.generate_causal_mask(max_len).to(x.device)
        
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Extract the output of the final REAL token, ignoring the padding
        last_token_indices = (seq_lengths - 1).view(-1, 1, 1).expand(-1, 1, self.d_model)
        last_out = torch.gather(out, 1, last_token_indices).squeeze(1)
        
        return self.fc_out(last_out)

# =========================================================================
# 2. DATASET EXTRACTION
# =========================================================================
class CinematicSequenceDataset(Dataset):
    def __init__(self, X_cont, lengths, y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X_cont)
    def __getitem__(self, idx): return self.X_cont[idx], self.lengths[idx], self.y[idx]

def extract_padded_sequences(df, feature_cols, target_col):
    """Slides a window over chronological conversations using RIGHT PADDING."""
    X_cont_padded = []
    lengths = []
    y_targets = []
    
    # Sort chronologically
    df = df.sort_values(by=['Conversation_ID', 'Turn_Number'])
    grouped = df.groupby('Conversation_ID')
    
    for _, group in grouped:
        features = group[feature_cols].values
        targets = group[target_col].values
        n_turns = len(group)
        
        for i in range(1, n_turns):
            history_cont = features[max(0, i - MAX_SEQ_LENGTH) : i]
            actual_len = len(history_cont)
            
            # Right Padding at the end of the sequence
            if actual_len < MAX_SEQ_LENGTH:
                pad_len = MAX_SEQ_LENGTH - actual_len
                pad_cont = np.zeros((pad_len, len(feature_cols)))
                history_cont = np.vstack([history_cont, pad_cont])
                
            X_cont_padded.append(history_cont)
            lengths.append(actual_len)
            y_targets.append(targets[i])
            
    return np.array(X_cont_padded), np.array(lengths), np.array(y_targets)

# =========================================================================
# 3. MAIN EXECUTION
# =========================================================================
def run_dual_transformer_trainer():
    print(f"\n{'='*80}")
    print(f"INITIALIZING V154: DUAL-DATASET CT-TRANSFORMER")
    print(f"{'='*80}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- LOAD DATASETS ---
    print("[SYSTEM] Loading Hollywood Cinematic Database...")
    df_hw = pd.read_csv(HOLLYWOOD_FILE, low_memory=False)
    df_hw['Dataset_Source'] = 'Hollywood'
    
    print("[SYSTEM] Loading DailyDialog Organic Database...")
    df_dd = pd.read_csv(DAILYDIALOG_FILE, low_memory=False)
    df_dd['Dataset_Source'] = 'DailyDialog'

    # Combine into one master frame for unified processing
    df = pd.concat([df_hw, df_dd], ignore_index=True)
    print(f"[SUCCESS] Combined Dataset: {len(df):,} total interaction nodes.")

    # We use pure continuous physics + Kinetic Velocities
    feature_cols = [f'Parent_{em}' for em in ALL_EMOTIONS] + ['Parent_X', 'Parent_Y', 'Parent_Z', 'Velocity_X', 'Velocity_Y', 'Velocity_Z']
    input_size = len(feature_cols)

    # --- EXTRACT TARGET COCKTAILS ---
    print("\n[SYSTEM] Extracting Dominant Emotional Cocktails across both domains...")
    child_cols = [f'Child_{em}' for em in ALL_EMOTIONS]
    child_data = df[child_cols].values
    
    top2_indices = np.argsort(child_data, axis=1)[:, -2:][:, ::-1]
    emotion_names = np.array([col.replace('Child_', '') for col in child_cols])
    df['Target_Cocktail'] = ['_'.join(labels) for labels in emotion_names[top2_indices]]

    # Keep the Top 40 most frequent reactions to prevent long-tail noise
    top_reactions = df['Target_Cocktail'].value_counts().head(40).index
    df = df[df['Target_Cocktail'].isin(top_reactions)].copy()
    
    le = LabelEncoder()
    df['Encoded_Target'] = le.fit_transform(df['Target_Cocktail'].values)
    num_classes = len(le.classes_)
    
    encoder_path = os.path.join(OUTPUT_DIR, "v154_dual_transformer_label_encoder.joblib")
    joblib.dump(le, encoder_path)

    # --- STRATIFIED 80/20 SPLIT (EQUAL PROPORTIONS) ---
    print("\n[SYSTEM] Applying perfectly balanced 80/20 split across both datasets...")
    
    df_hw_filtered = df[df['Dataset_Source'] == 'Hollywood']
    df_dd_filtered = df[df['Dataset_Source'] == 'DailyDialog']
    
    # Split Hollywood Convos
    hw_train_convos, hw_test_convos = train_test_split(df_hw_filtered['Conversation_ID'].unique(), test_size=0.20, random_state=42)
    # Split DailyDialog Convos
    dd_train_convos, dd_test_convos = train_test_split(df_dd_filtered['Conversation_ID'].unique(), test_size=0.20, random_state=42)

    train_convos = np.concatenate([hw_train_convos, dd_train_convos])
    test_convos = np.concatenate([hw_test_convos, dd_test_convos])

    df_train = df[df['Conversation_ID'].isin(train_convos)].copy()
    df_test = df[df['Conversation_ID'].isin(test_convos)].copy()

    # --- SEQUENCE EXTRACTION ---
    print(f"[SYSTEM] Extracting Transformer Memory Sequences (Right-Padded Window = {MAX_SEQ_LENGTH})...")
    X_cont_train, lengths_train, y_train = extract_padded_sequences(df_train, feature_cols, 'Encoded_Target')
    X_cont_test, lengths_test, y_test = extract_padded_sequences(df_test, feature_cols, 'Encoded_Target')
    
    print(f"[SUCCESS] {len(X_cont_train):,} training sequences and {len(X_cont_test):,} testing sequences mapped.")

    train_loader = DataLoader(CinematicSequenceDataset(X_cont_train, lengths_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CinematicSequenceDataset(X_cont_test, lengths_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[SYSTEM] Booting PyTorch Multi-Head Attention Engine on {str(device).upper()}...")

    # --- TRAINING THE DUAL-DOMAIN TRANSFORMER ---
    model = CinematicDirectorTransformer(input_size, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n[TRAINING THE MASTER DIRECTOR (HOLLYWOOD + DAILYDIALOG)]")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X_cont, batch_lengths, batch_y in train_loader:
            batch_X_cont, batch_lengths, batch_y = batch_X_cont.to(device), batch_lengths.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X_cont, batch_lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f" -> Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f}")

    # --- EVALUATION ---
    print("\n[SYSTEM] Evaluating Transformer on Unseen Hollywood and Organic Data...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X_cont, batch_lengths, batch_y in test_loader:
            batch_X_cont, batch_lengths, batch_y = batch_X_cont.to(device), batch_lengths.to(device), batch_y.to(device)
            outputs = model(batch_X_cont, batch_lengths)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    accuracy = accuracy_score(all_targets, all_preds) * 100
    
    print(f"\n{'='*80}")
    print(f"THE DUAL-DOMAIN TRANSFORMER (V154) VALIDATION REPORT")
    print(f"{'='*80}")
    
    print("CLASSIFICATION BREAKDOWN (Top 40 Cocktails):")
    target_names_str = le.inverse_transform(range(num_classes))
    report = classification_report(all_targets, all_preds, target_names=target_names_str, zero_division=0)
    print(report)
    
    print("-" * 80)
    print(f"Total Combined Scenes Evaluated : {len(X_cont_test):,}")
    print(f"True Predictive Accuracy        : {accuracy:.2f}%")
    print(f"{'='*80}\n")
    
    model_path = os.path.join(OUTPUT_DIR, "v154_dual_ct_transformer_director.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Multi-Domain Neural Brain saved to:\n -> {model_path}")

if __name__ == '__main__':
    run_dual_transformer_trainer()