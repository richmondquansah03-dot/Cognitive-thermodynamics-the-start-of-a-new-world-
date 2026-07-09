import tkinter as tk
from tkinter import scrolledtext, font
import threading
import torch
import torch.nn as nn
from transformers import pipeline
import requests
import joblib
import math
import time
import os
import re
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# V160: DYNAMIC LIMBIC CHATBOT
# Upgrades: Composite Attention, Neuroplastic Learning, Goal Injection, Telemetry
# =========================================================================

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:text"  # Base model for unchained autocomplete

# --- ASSETS ---
MODELS_DIR = r"C:\Users\GIGABYTE\Documents\ct\Ouroboros_Outputs\AEI_Generative"
LCTM_WEIGHTS = os.path.join(MODELS_DIR, "v154_dual_ct_transformer_director.pt")
LABEL_ENCODER = os.path.join(MODELS_DIR, "v154_dual_transformer_label_encoder.joblib")

# --- HYPERPARAMETERS ---
CANDIDATES_PER_TURN = 5   
MAX_SEQ_LENGTH = 16   
D_MODEL = 256         
N_HEAD = 8            
NUM_LAYERS = 3        
DIM_FEEDFORWARD = 512 

OBSERVER_LABELS = ['curiosity', 'realization', 'confusion', 'admiration', 'excitement', 'optimism', 'gratitude', 'remorse', 'desire']
ADMIN_LABELS = ['anger', 'annoyance', 'disapproval', 'disgust', 'fear', 'nervousness', 'approval', 'caring', 'pride', 'joy', 'love', 'sadness', 'relief']
EXTERNAL_CHANGE_LABELS = ['disappointment', 'embarrassment', 'grief']
NEUTRAL_LABEL = ['neutral']
MODIFIER_LABELS = ['amusement', 'surprise']

ALL_EMOTIONS = OBSERVER_LABELS + ADMIN_LABELS + EXTERNAL_CHANGE_LABELS + NEUTRAL_LABEL + MODIFIER_LABELS
INPUT_SIZE = len(ALL_EMOTIONS) + 6 # 28 emotions + X, Y, Z + Vel_X, Vel_Y, Vel_Z

class DigitalLimbicSystem:
    def __init__(self):
        # Baseline interests (Can grow over time)
        self.known_interests = {'anime', 'game', 'rpg', 'goku', 'manga', 'playstation', 'pc', 'boss', 'level', 'zelda', 'cyberpunk', 'sci-fi', 'movie', 'marvel', 'tech', 'coding'}
        self.relationship_score = 0.0 # Ranges from -1.0 to 1.0
        self.topic_memory = [] # Tracks recent vocabulary to measure novelty
        
    def calculate_attention(self, text_input, user_dna):
        """
        Attention = Emotion + Interest + Novelty + Relationship
        """
        words = set(re.findall(r'\b\w+\b', text_input.lower()))
        
        # 1. Emotional Intensity (How far from neutral is the user?)
        emotion_intensity = 1.0 - user_dna.get('neutral', 1.0)
        
        # 2. Interest Match
        interest_score = 1.0 if any(w in self.known_interests for w in words) else 0.0
        
        # 3. Novelty (Inverse of recent topic overlap)
        novelty = 1.0
        if self.topic_memory:
            recent_words = set().union(*self.topic_memory)
            overlap = len(words.intersection(recent_words))
            novelty = max(0.0, 1.0 - (overlap / (len(words) + 0.1)))
        
        # Update topic memory
        self.topic_memory.append(words)
        if len(self.topic_memory) > 3: 
            self.topic_memory.pop(0)
            
        # 4. Relationship Update
        pos_em = sum([user_dna.get(e, 0.0) for e in ['admiration', 'approval', 'joy', 'gratitude', 'love']])
        neg_em = sum([user_dna.get(e, 0.0) for e in ['anger', 'annoyance', 'disapproval', 'disgust']])
        self.relationship_score = max(-1.0, min(1.0, self.relationship_score + (pos_em - neg_em) * 0.1))
        
        # --- THE COMPOSITE FORMULA ---
        attention = (0.4 * emotion_intensity) + (0.3 * interest_score) + (0.2 * novelty) + (0.1 * ((self.relationship_score + 1.0) / 2.0))
        attention = max(0.1, min(attention + 0.2, 1.0)) # Floor at 0.1, slight baseline boost
        
        # --- NEUROPLASTICITY (LEARNING) ---
        # If the user is highly passionate/excited, the agent learns to care about those words
        if user_dna.get('excitement', 0) + user_dna.get('joy', 0) + user_dna.get('curiosity', 0) > 0.5:
            for w in words:
                if len(w) > 4: # Ignore short words like "the", "and"
                    self.known_interests.add(w)
                    
        # State Descriptor
        if attention > 0.75:
            state_desc = "Highly engaged, processing intent actively, leaning in."
        elif attention > 0.45:
            state_desc = "Casual listening, responding normally."
        else:
            state_desc = "Aloof, distracted, avoiding deep engagement."
            
        return attention, state_desc, emotion_intensity, interest_score, novelty

def extract_conversational_goal(text):
    """Simple heuristic to ensure the Base LLM doesn't ignore the user's intent."""
    text_lower = text.lower()
    if "?" in text:
        if any(w in text_lower for w in ['what', 'who', 'where', 'when', 'why', 'how']):
            return "Provide a direct, factual answer to the specific question asked."
        else:
            return "Answer the yes/no question clearly."
    elif any(w in text_lower for w in ['help', 'advice', 'pointers', 'tips', 'recommend']):
        return "Offer concrete advice, examples, or actionable recommendations."
    else:
        return "Acknowledge the statement and keep the conversation flowing naturally."

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

def get_pure_physics(text, emotion_sensor, prev_x=0.0, prev_y=0.0, prev_z=0.0):
    with torch.no_grad():
        results = emotion_sensor(text)[0]
        
    scores = {item['label']: item['score'] for item in results}
    total_prob = sum(scores.values()) if sum(scores.values()) > 0 else 1.0
    
    dna = {em: scores.get(em, 0.0) / total_prob for em in ALL_EMOTIONS}
    
    p_obs = sum(dna[em] for em in OBSERVER_LABELS)
    p_adm = sum(dna[em] for em in ADMIN_LABELS)
    p_ext = sum(dna[em] for em in EXTERNAL_CHANGE_LABELS)
    p_neu = dna['neutral']
    p_amu = dna['amusement']
    p_sur = dna['surprise']
    
    V_Obs = p_obs * 10.0
    V_Adm = p_adm * 1.2 * 10.0 
    V_Ext = p_ext * 1.5 * 10.0 
    
    Z = min(p_neu * 10.0, 10.0)
    Y = min(V_Ext, 10.0)
    X = (V_Obs - V_Adm) + (p_sur * 10.0) - (p_amu * 10.0)
    
    vel_x = X - prev_x
    vel_y = Y - prev_y
    vel_z = Z - prev_z
    
    feature_vector = [dna[em] for em in ALL_EMOTIONS] + [X, Y, Z, vel_x, vel_y, vel_z]
    
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = [f"{e} ({s*100:.1f}%)" for e, s in sorted_emotions[:3]]
    
    return feature_vector, dna, top_3, X, Y, Z

def generate_base_candidates(script_context, target_emotions, attention_alpha, focus_desc, goal):
    """
    Prompts the Base Model using a screenplay format, now including explicit conversational goals.
    """
    emotion_str = ", ".join([e.capitalize() for e in target_emotions])
    
    prompt = f"""[SCENE START]
{script_context}
[David's Internal State: Attention = {attention_alpha:.2f}. {focus_desc}]
[David's Objective: {goal}]
David: ({emotion_str})"""

    candidates = []
    
    for _ in range(CANDIDATES_PER_TURN):
        try:
            payload = {
                "model": "llama3:text" if "llama3:text" in OLLAMA_MODEL else "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.85, 
                    "stop": ["\n", "[SCENE", "User:", "["] 
                }
            }
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            
            raw_text = response.json()['response'].strip()
            clean_text = raw_text.strip(' \n\t\r."\'')
            
            if clean_text:
                candidates.append(clean_text)
                
        except Exception as e:
            pass
            
    return candidates if candidates else ["..."]

class OuroborosAttentionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("V160: Dynamic Limbic Chatbot")
        self.root.geometry("1400x850")
        self.root.configure(bg="#0f172a")

        self.conversation_history = []
        self.physics_history = []
        
        self.last_x, self.last_y, self.last_z = 0.0, 0.0, 10.0
        self.limbic_system = DigitalLimbicSystem()

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        chat_font = font.Font(family="Consolas", size=11)
        telemetry_font = font.Font(family="Consolas", size=10)
        
        left_frame = tk.Frame(self.root, bg="#0f172a")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(left_frame, text="INTERACTION TERMINAL", bg="#0f172a", fg="#0ea5e9", font=("Arial", 14, "bold")).pack(anchor=tk.W)
        
        self.chat_display = scrolledtext.ScrolledText(left_frame, bg="#1e293b", fg="white", font=chat_font, wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_display.config(state=tk.DISABLED)

        input_frame = tk.Frame(left_frame, bg="#0f172a")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_box = tk.Entry(input_frame, bg="#334155", fg="white", font=("Arial", 12), insertbackground="white")
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.input_box.bind("<Return>", lambda event: self.handle_user_input())
        
        self.send_btn = tk.Button(input_frame, text="SEND", bg="#0ea5e9", fg="white", font=("Arial", 10, "bold"), command=self.handle_user_input)
        self.send_btn.pack(side=tk.RIGHT, padx=5, ipadx=10)

        right_frame = tk.Frame(self.root, bg="#0f172a", width=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="SYSTEM TELEMETRY (THE 6 BRAINS)", bg="#0f172a", fg="#facc15", font=("Arial", 14, "bold")).pack(anchor=tk.W)
        
        self.telemetry_display = scrolledtext.ScrolledText(right_frame, bg="#1e293b", fg="#a7f3d0", font=telemetry_font, wrap=tk.WORD)
        self.telemetry_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.telemetry_display.config(state=tk.DISABLED)

    def log_chat(self, speaker, text, color):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{speaker}: ", color)
        self.chat_display.insert(tk.END, f"{text}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        self.chat_display.tag_config("user", foreground="#0ea5e9")
        self.chat_display.tag_config("ai", foreground="#facc15")

    def log_telemetry(self, text, color=None):
        self.telemetry_display.config(state=tk.NORMAL)
        if color:
            # Generate unique tag for this specific insert to color it
            tag_name = f"color_{time.time()}"
            self.telemetry_display.insert(tk.END, f"{text}\n", tag_name)
            self.telemetry_display.tag_config(tag_name, foreground=color)
        else:
            self.telemetry_display.insert(tk.END, f"{text}\n")
        self.telemetry_display.see(tk.END)
        self.telemetry_display.config(state=tk.DISABLED)

    def load_models(self):
        self.log_telemetry("[BRAIN 4] Waking up RoBERTa (Verifier) on GPU...")
        self.root.update()
        threading.Thread(target=self._async_load_models).start()

    def _async_load_models(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.emotion_sensor = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=0 if torch.cuda.is_available() else -1, truncation=True)
            
            self.log_telemetry("[BRAIN 2] Waking up V154 Dual-Dataset LCTM...")
            self.le = joblib.load(LABEL_ENCODER)
            num_classes = len(self.le.classes_)
            
            self.director_model = CinematicDirectorTransformer(INPUT_SIZE, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, num_classes).to(self.device)
            self.director_model.load_state_dict(torch.load(LCTM_WEIGHTS, map_location=self.device))
            self.director_model.eval()
            
            self.log_telemetry("[SUCCESS] System Online. Waiting for User Input.\n", "#4ade80")
        except Exception as e:
            self.log_telemetry(f"[ERROR] Failed to load models: {e}", "#f43f5e")

    def handle_user_input(self):
        user_text = self.input_box.get().strip()
        if not user_text: return
        
        self.input_box.delete(0, tk.END)
        self.input_box.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        
        self.log_chat("User", user_text, "user")
        
        threading.Thread(target=self.process_turn, args=(user_text,)).start()

    def process_turn(self, user_text):
        self.log_telemetry("\n" + "="*50)
        
        # 1. Physics Extraction & Goal Parsing
        vec, dna, top_3, nx, ny, nz = get_pure_physics(user_text, self.emotion_sensor, self.last_x, self.last_y, self.last_z)
        self.last_x, self.last_y, self.last_z = nx, ny, nz
        
        user_goal = extract_conversational_goal(user_text)
        
        self.physics_history.append(vec)
        self.conversation_history.append({"speaker": "User", "text": user_text})
        self.log_telemetry(f"User Physics: {top_3}")
        
        # 2. Composite Attention State (The Limbic Engine)
        alpha, focus_desc, e_int, i_score, nov = self.limbic_system.calculate_attention(user_text, dna)
        
        self.log_telemetry(f"[BRAIN 6 | LIMBIC] Calculating Attention Formula:", "#0ea5e9")
        self.log_telemetry(f" -> Emotion ({e_int:.2f}) + Interest ({i_score:.2f}) + Novelty ({nov:.2f})")
        self.log_telemetry(f" -> Relationship Score: {self.limbic_system.relationship_score:.2f}")
        
        if alpha > 0.6:
            self.log_telemetry(f" -> FINAL ATTENTION: {alpha:.2f} [{focus_desc}]", "#facc15") 
        else:
            self.log_telemetry(f" -> FINAL ATTENTION: {alpha:.2f} [{focus_desc}]", "#94a3b8") 

        # 3. LCTM Target Prediction
        history_window = self.physics_history[-MAX_SEQ_LENGTH:]
        actual_len = len(history_window)
        
        if actual_len < MAX_SEQ_LENGTH:
            pad_len = MAX_SEQ_LENGTH - actual_len
            padding = [[0.0] * INPUT_SIZE for _ in range(pad_len)]
            history_window = history_window + padding
            
        tensor_window = torch.tensor([history_window], dtype=torch.float32).to(self.device)
        seq_lengths = torch.tensor([actual_len], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.director_model(tensor_window, seq_lengths)
            prediction_idx = torch.argmax(output, 1).item()
            
        target_cocktail_str = self.le.inverse_transform([prediction_idx])[0]
        target_emotions = target_cocktail_str.split('_')
        self.log_telemetry(f"[BRAIN 2 | LCTM] Predicts State: {target_emotions}")
        self.log_telemetry(f"[BRAIN 2 | LCTM] Extracted Goal: {user_goal}")

        # 4. Generate Candidates
        script_context = "\n".join([f"{s['speaker']}: {s['text']}" for s in self.conversation_history])
        self.log_telemetry(f"[BRAIN 3 | LLM] Autocompleting variants...")
        candidates = generate_base_candidates(script_context, target_emotions, alpha, focus_desc, user_goal)
        
        # 5. Verify and Optimize (Brain 4 & 5)
        best_take = None
        best_score = -1.0
        best_vec = None
        best_nx, best_ny, best_nz = self.last_x, self.last_y, self.last_z
        
        self.log_telemetry(f"[BRAIN 4 & 5] Verification & Optimization Loop:")
        for i, cand in enumerate(candidates):
            vec, dna, top_3, nx, ny, nz = get_pure_physics(cand, self.emotion_sensor, self.last_x, self.last_y, self.last_z)
            cocktail_score = sum([dna.get(em, 0.0) for em in target_emotions])
            
            # Log every candidate's score to prove the optimization loop is working
            self.log_telemetry(f"  Cand {i+1} [{cocktail_score*100:>4.1f}% Match] : {cand[:40]}...")
            
            if cocktail_score > best_score:
                best_score = cocktail_score
                best_take = cand
                best_vec = vec
                best_nx, best_ny, best_nz = nx, ny, nz

        self.last_x, self.last_y, self.last_z = best_nx, best_ny, best_nz
        
        self.log_telemetry(f"-> WINNER SELECTED: Candidate Match {best_score*100:.1f}%", "#4ade80")
        
        self.log_chat("David", best_take, "ai")
        self.conversation_history.append({"speaker": "David", "text": best_take})
        self.physics_history.append(best_vec)

        self.input_box.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)
        self.input_box.focus()

if __name__ == "__main__":
    root = tk.Tk()
    app = OuroborosAttentionUI(root)
    root.mainloop()