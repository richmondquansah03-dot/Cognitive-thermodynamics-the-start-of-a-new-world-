import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# OUROBOROS CYBER V7: THE ULTIMATE CYBER LCTM
# Fusing the Raw Network "Dictionary" with XYZ Physics for Multi-Class Threat Prediction
# =========================================================================

INPUT_FILE = r"C:\Users\GIGABYTE\Documents\ct\cybersecurity LCTM\cicids2017_binary_balanced.csv"

# Hyperparameters
MAX_SEQ_LENGTH = 10
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
BATCH_SIZE = 256
EPOCHS = 10 
LEARNING_RATE = 0.001

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

class CyberPredictorTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout=0.2):
        super().__init__()
        self.d_model = d_model
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
        arange_tensor = torch.arange(max_len, device=x_cont.device).unsqueeze(0).expand(batch_size, max_len)
        lengths_tensor = seq_lengths.unsqueeze(1).expand(batch_size, max_len)
        padding_mask = arange_tensor >= lengths_tensor
        
        x = self.input_projection(x_cont) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        causal_mask = self.generate_causal_mask(max_len).to(x.device)
        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        last_token_indices = (seq_lengths - 1).view(-1, 1, 1).expand(-1, 1, self.d_model)
        last_out = torch.gather(out, 1, last_token_indices).squeeze(1)
        return self.fc_out(last_out)

class CyberSequenceDataset(Dataset):
    def __init__(self, X_cont, lengths, y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X_cont)
    def __getitem__(self, idx): return self.X_cont[idx], self.lengths[idx], self.y[idx]

def extract_padded_sequences(df, feature_cols, target_col):
    X_padded, lengths, y_targets = [], [], []
    grouped = df.groupby('Destination Port')
    
    for port, group in grouped:
        features = group[feature_cols].values
        targets = group[target_col].values
        n_turns = len(group)
        for i in range(1, n_turns):
            history_cont = features[max(0, i - MAX_SEQ_LENGTH) : i]
            actual_len = len(history_cont)
            if actual_len < MAX_SEQ_LENGTH:
                pad_len = MAX_SEQ_LENGTH - actual_len
                pad_cont = np.zeros((pad_len, len(feature_cols)))
                history_cont = np.vstack([history_cont, pad_cont])
                
            X_padded.append(history_cont)
            lengths.append(actual_len)
            y_targets.append(targets[i])
            
    return np.array(X_padded), np.array(lengths), np.array(y_targets)

def run_v7_ultimate_lctm():
    print(f"\n{'='*80}")
    print("INITIALIZING V7: THE ULTIMATE CYBER LCTM (RAW DICTIONARY + XYZ PHYSICS)")
    print(f"{'='*80}")

    print("[SYSTEM] Loading Dataset...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print("[CRITICAL ERROR] File not found.")
        return

    # We want to predict the EXACT attack name, not just 0 or 1.
    # We look for 'Label' first. If it doesn't exist, we fall back to Attack_Binary.
    target_col = 'Label' if 'Label' in df.columns else 'Attack_Binary'
    
    # Encode the string labels (e.g., 'BENIGN', 'PortScan', 'DoS Hulk') into integers
    le = LabelEncoder()
    df['Encoded_Target'] = le.fit_transform(df[target_col].astype(str))
    num_classes = len(le.classes_)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = MinMaxScaler()

    # --- 1. THE RAW NETWORK DICTIONARY (Equivalent to the 28 Emotions) ---
    raw_cols = [
        'Flow Packets/s', 'Fwd Packet Length Mean', 'Fwd Packet Length Max', 'Packet Length Variance',
        'Total Bwd Packets', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 
        'ACK Flag Count', 'FIN Flag Count', 'PSH Flag Count', 'SYN Flag Count', 'URG Flag Count'
    ]
    
    for col in raw_cols:
        if col in df.columns:
            df[f'raw_{col}'] = scaler.fit_transform(df[[col]])
        else:
            df[f'raw_{col}'] = 0.0
            
    raw_feature_set = [f'raw_{c}' for c in raw_cols]

    # --- 2. THE XYZ THERMODYNAMIC PHYSICS ---
    print("[SYSTEM] Engineering Thermodynamic Threat Geometry...")
    df['Protocol_Violation'] = (df['raw_FIN Flag Count'] + df['raw_PSH Flag Count'] + df['raw_URG Flag Count'] + df['raw_SYN Flag Count']) * (1.0 - df['raw_ACK Flag Count'])
    
    df['Req_Recon_X'] = (df['raw_Flow Packets/s'] * (1.0 - df['raw_Fwd Packet Length Mean'])) * 20.0
    df['Req_Exploit_Y'] = (df['raw_Fwd Packet Length Max'] + df['raw_Packet Length Variance'] + df['Protocol_Violation']) * 50.0
    df['Req_Neutral_Z'] = (df['raw_ACK Flag Count'] * 10.0) - df['Req_Recon_X'] - df['Req_Exploit_Y']
    
    df['Srv_Receptive_X'] = df['raw_Total Bwd Packets'] * 10.0
    df['Srv_Defensive_X'] = ((1.0 - df['raw_Total Bwd Packets']) + df['raw_FIN Flag Count']) * 10.0
    df['Srv_Exfil_Y'] = df['raw_Bwd Packet Length Max'] * 50.0
    df['Srv_Neutral_Z'] = df['raw_Bwd Packet Length Mean'] * 10.0

    df['Requester_X'] = df['Req_Recon_X'].clip(lower=0.0, upper=10.0)
    df['Requester_Y'] = df['Req_Exploit_Y'].clip(lower=0.0, upper=10.0)
    df['Requester_Z'] = df['Req_Neutral_Z'].clip(lower=0.0, upper=10.0)
    df['Server_X'] = (df['Srv_Receptive_X'] - df['Srv_Defensive_X']).clip(lower=-10.0, upper=10.0)
    df['Server_Y'] = df['Srv_Exfil_Y'].clip(lower=0.0, upper=10.0)
    df['Server_Z'] = df['Srv_Neutral_Z'].clip(lower=0.0, upper=10.0)
    df['Flow_Duration_Log'] = np.log1p(df['Flow Duration'].clip(lower=0))

    ct_feature_set = ['Flow_Duration_Log', 'Protocol_Violation', 'Requester_X', 'Requester_Y', 'Requester_Z', 'Server_X', 'Server_Y', 'Server_Z']

    # --- 3. THE UNIFIED FUSION MATRIX ---
    # This matches the exact design of the Hollywood LCTM (Emotions + Physics)
    unified_feature_set = raw_feature_set + ct_feature_set
    input_size = len(unified_feature_set)

    print("\n[SYSTEM] Extracting Temporal Sequences using the Unified Matrix...")
    X_unified, len_unified, y_unified = extract_padded_sequences(df, unified_feature_set, 'Encoded_Target')

    idx_train, idx_test = train_test_split(np.arange(len(y_unified)), test_size=0.20, random_state=42, stratify=y_unified)

    train_loader = DataLoader(CyberSequenceDataset(X_unified[idx_train], len_unified[idx_train], y_unified[idx_train]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CyberSequenceDataset(X_unified[idx_test], len_unified[idx_test], y_unified[idx_test]), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[SYSTEM] Booting the Master LCTM on {str(device).upper()}...")

    model = CyberPredictorTransformer(input_size, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    class_counts = np.bincount(y_unified[idx_train])
    weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"\n[TRAINING MULTI-CLASS LCTM - {num_classes} THREAT PROFILES]")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for bX, blen, by in train_loader:
            bX, blen, by = bX.to(device), blen.to(device), by.to(device)
            opt.zero_grad()
            out = model(bX, blen)
            loss = criterion(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        print(f" -> Epoch [{epoch+1}/{EPOCHS}] | Unified Loss: {total_loss/len(train_loader):.4f}")

    print("\n[SYSTEM] Evaluating Multi-Class Predictive Radar...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bX, blen, by in test_loader:
            bX, blen, by = bX.to(device), blen.to(device), by.to(device)
            out = model(bX, blen)
            _, predicted = torch.max(out.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(by.cpu().numpy())

    print(f"\n{'='*80}")
    print(f"THE CYBER LCTM (V7): THREAT DICTIONARY VALIDATION REPORT")
    print(f"{'='*80}")
    
    target_names_str = le.inverse_transform(range(num_classes))
    report = classification_report(all_targets, all_preds, target_names=target_names_str, zero_division=0)
    print(report)
    
    accuracy = accuracy_score(all_targets, all_preds) * 100
    print("-" * 80)
    print(f"Total Sequences Evaluated : {len(X_unified[idx_test]):,}")
    print(f"Predictive Accuracy       : {accuracy:.2f}%")
    print(f"Architecture Profile      : {len(raw_feature_set)} Dictionary Nodes + {len(ct_feature_set)} Thermodynamic Physics")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    run_v7_ultimate_lctm()