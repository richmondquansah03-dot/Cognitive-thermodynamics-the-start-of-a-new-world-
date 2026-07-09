import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import os
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# OUROBOROS CYBER V3: DUAL-KINETIC EXTRACTOR
# Implementing the Zero-Day Subtraction Protocol & Gravitational Multipliers
# =========================================================================

# -> Point this to your CICIDS2017 Balanced CSV
INPUT_FILE = r"C:\Users\GIGABYTE\Documents\ct\cybersecurity LCTM\cicids2017_binary_balanced.csv"
OUTPUT_DIR = r"C:\Users\GIGABYTE\Documents\ct\cybersecurity LCTM\Ouroboros_Cyber_Outputs"

# The Gravitational Multiplier (Yanks anomalies out of the noise)
ANOMALY_GRAVITY = 50.0 
RECON_GRAVITY = 20.0

def run_v3_extractor():
    print(f"\n{'='*80}")
    print("INITIALIZING CYBER V3: DUAL-KINETIC EXTRACTOR (ZERO-DAY PATCHED)")
    print(f"{'='*80}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[SYSTEM] Loading CICIDS2017 Network Flow Matrix...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"[CRITICAL ERROR] File not found: {INPUT_FILE}")
        return

    # Standardize the label column
    label_col = 'Attack_Binary' if 'Attack_Binary' in df.columns else 'Label'
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    TOTAL_FLOWS = len(df)
    
    print(f"[SUCCESS] Loaded {TOTAL_FLOWS:,} network conversations.\n")

    print("[PHASE 1] Normalizing structural flow arrays...")
    scaler = MinMaxScaler()
    
    # Safely select columns that exist in the dataset
    potential_cols = [
        'Flow Packets/s', 'Fwd Packet Length Mean', 'Fwd Packet Length Max', 'Packet Length Variance',
        'Total Bwd Packets', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 
        'ACK Flag Count', 'FIN Flag Count', 'PSH Flag Count', 'SYN Flag Count', 'URG Flag Count'
    ]
    
    # Bulletproof loop: If the column is missing, safely default it to 0.0
    for col in potential_cols:
        if col in df.columns:
            df[f'scaled_{col}'] = scaler.fit_transform(df[[col]])
        else:
            df[f'scaled_{col}'] = 0.0

    print("[PHASE 2] Executing Zero-Day Subtraction Protocol...")
    
    # ---------------------------------------------------------
    # THE SUBTRACTION PROTOCOL (TCP FLAG ANOMALY)
    # ---------------------------------------------------------
    # In standard TCP, ACK is the baseline. 
    # If a flow contains PSH, FIN, URG, or SYN without proportional ACKs, it is a protocol violation.
    illegal_flags = df['scaled_FIN Flag Count'] + df['scaled_PSH Flag Count'] + df['scaled_URG Flag Count'] + df['scaled_SYN Flag Count']
    
    # If illegal flags exist but ACK is low/missing, the anomaly score spikes.
    df['Protocol_Violation'] = illegal_flags * (1.0 - df['scaled_ACK Flag Count'])

    print("[PHASE 3] Extracting Dual-Signature Kinematics (Parent vs Child)...")

    # --- THE REQUESTER (CLIENT) ---
    # Recon (+X): High packet rate, tiny payload. Accelerated by Recon Gravity.
    df['Req_Recon_X'] = (df['scaled_Flow Packets/s'] * (1.0 - df['scaled_Fwd Packet Length Mean'])) * RECON_GRAVITY
    
    # Exploit (Y): The Payload + The Protocol Violation. Multiplied by massive Anomaly Gravity.
    # This ensures 0-day exploits (high Protocol_Violation, low payload) trigger the Y axis!
    raw_y = df['scaled_Fwd Packet Length Max'] + df['scaled_Packet Length Variance'] + df['Protocol_Violation']
    df['Req_Exploit_Y'] = raw_y * ANOMALY_GRAVITY
    
    # Neutral (Z): Whatever energy is left over, heavily reliant on standard ACK flags.
    df['Req_Neutral_Z'] = (df['scaled_ACK Flag Count'] * 10.0) - df['Req_Recon_X'] - df['Req_Exploit_Y']

    # --- THE SERVER (HOST) ---
    # Receptive (+X): Server is answering back (Standard Routing)
    df['Srv_Receptive_X'] = df['scaled_Total Bwd Packets'] * 10.0
    
    # Defensive (-X): Server goes silent, or drops connection (FIN/RST flags)
    df['Srv_Defensive_X'] = ((1.0 - df['scaled_Total Bwd Packets']) + df['scaled_FIN Flag Count']) * 10.0
    
    # Hemorrhage (Y): Server is exfiltrating massive data (Data theft)
    df['Srv_Exfil_Y'] = df['scaled_Bwd Packet Length Max'] * ANOMALY_GRAVITY
    
    # Neutral (Z): Sending back standard-sized, acknowledged responses.
    df['Srv_Neutral_Z'] = df['scaled_Bwd Packet Length Mean'] * 10.0

    print("[PHASE 4] Applying Equilibrium Bounds (0.0 to 10.0)...")

    # Final Requester Coordinates
    df['Requester_X'] = df['Req_Recon_X'].clip(lower=0.0, upper=10.0)
    df['Requester_Y'] = df['Req_Exploit_Y'].clip(lower=0.0, upper=10.0)
    df['Requester_Z'] = df['Req_Neutral_Z'].clip(lower=0.0, upper=10.0)

    # Final Server Coordinates (Net X = Receptive - Defensive)
    # Note: Server X goes from -10 to 10. (-10 is maximum lockdown, +10 is fully open).
    df['Server_X'] = (df['Srv_Receptive_X'] - df['Srv_Defensive_X']).clip(lower=-10.0, upper=10.0)
    df['Server_Y'] = df['Srv_Exfil_Y'].clip(lower=0.0, upper=10.0)
    df['Server_Z'] = df['Srv_Neutral_Z'].clip(lower=0.0, upper=10.0)

    print("\n[SYSTEM] Securing V3 Dual-Kinetic Master Database...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    master_path = os.path.join(OUTPUT_DIR, f"V3_DUAL_KINETIC_MASTER_{timestamp}.csv")
    
    cols_to_save = [
        'Destination Port', 'Flow Duration', label_col, 'Protocol_Violation',
        'Requester_X', 'Requester_Y', 'Requester_Z',
        'Server_X', 'Server_Y', 'Server_Z'
    ]
    
    final_cols = [c for c in cols_to_save if c in df.columns]
    df[final_cols].to_csv(master_path, index=False)
    
    print(f"\n[SUCCESS] V3 Subtraction Engine execution complete!")
    print(f" -> Gravity Multipliers Applied : Y-Axis ({ANOMALY_GRAVITY}x), X-Axis ({RECON_GRAVITY}x)")
    print(f" -> [SAVED] {master_path}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    run_v3_extractor()