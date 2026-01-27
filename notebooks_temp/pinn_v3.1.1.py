import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sgp4.api import Satrec, WGS72

# --- 1. SYSTEM CONFIGURATION ---
CONFIG = {
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'R_REF': 6378.137,       # Earth Radius (km)
    'SEED': 42,              # Deterministic Training
    'EPOCHS': 150,
    'LR': 0.001,
    'MODEL_PATH': 'pinn_model.pth',
    'SCALER_PATH': 'pinn_scalers.pkl',
    'DATA_PATH': 'training_residuals.csv'
}

# --- 2. REPRODUCIBILITY ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['SEED'])

# --- 3. MODEL ARCHITECTURE ---
class GatedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 9 Features (rx, ry, rz, vx, vy, vz, bstar, ndot, dt)
        self.net = nn.Sequential(
            nn.Linear(9, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 6) # Output: Correction (dr, dv)
        )

    def forward(self, x, t):
        # x shape: [Batch, 9] (Scaled features)
        # t shape: [Batch, 1] (Normalized time)
        out = self.net(x)
        
        # Physics Gate: Forces Correction=0 at t=0
        # tanh(5*t) rises quickly, "opening" the gate within ~1-2 hours
        gate = torch.tanh(5.0 * t) 
        return out * gate

# --- 4. ORBIT DEPLOYER CLASS (The Brain) ---
class OrbitDeployer:
    def __init__(self):
        self.model = GatedPINN().to(CONFIG['DEVICE'])
        self.scaler_X = None
        self.scaler_y = None
        
    def train(self, csv_path):
        print(f"[-] Training Gated PINN from {csv_path}...")
        
        # A. Load & Preprocess
        df = pd.read_csv(csv_path)
        X_raw = df[['input_rx', 'input_ry', 'input_rz', 'input_vx', 'input_vy', 'input_vz', 
                    'bstar', 'ndot', 'dt_minutes']].values.astype(np.float32)
        y_raw = df[['err_rx', 'err_ry', 'err_rz', 'err_vx', 'err_vy', 'err_vz']].values.astype(np.float32)
        
        # B. Fit Scalers
        self.scaler_X = StandardScaler().fit(X_raw)
        self.scaler_y = StandardScaler().fit(y_raw)
        
        # C. Prepare Tensor Data
        X_scaled = self.scaler_X.transform(X_raw)
        y_scaled = self.scaler_y.transform(y_raw)
        t_gate = torch.tensor(df['dt_minutes'].values / 1440.0, dtype=torch.float32).unsqueeze(1)
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled), torch.tensor(y_scaled), t_gate
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # D. Training Loop
        optimizer = optim.Adam(self.model.parameters(), lr=CONFIG['LR'])
        loss_fn = nn.MSELoss()
        
        self.model.train()
        for epoch in range(CONFIG['EPOCHS']):
            for batch_x, batch_y, batch_t in loader:
                batch_x, batch_y, batch_t = batch_x.to(CONFIG['DEVICE']), batch_y.to(CONFIG['DEVICE']), batch_t.to(CONFIG['DEVICE'])
                
                pred = self.model(batch_x, batch_t)
                loss = loss_fn(pred, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        print("[+] Training Complete.")
        self.save_artifacts()

    def save_artifacts(self):
        # Save Weights
        torch.save(self.model.state_dict(), CONFIG['MODEL_PATH'])
        # Save Scalers
        joblib.dump({'X': self.scaler_X, 'y': self.scaler_y}, CONFIG['SCALER_PATH'])
        print(f"[+] Model saved to {CONFIG['MODEL_PATH']}")

    def load_artifacts(self):
        if not os.path.exists(CONFIG['MODEL_PATH']):
            raise FileNotFoundError("Model not found. Run .train() first.")
            
        print("[-] Loading model weights and scalers...")
        self.model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
        self.model.eval()
        
        scalers = joblib.load(CONFIG['SCALER_PATH'])
        self.scaler_X = scalers['X']
        self.scaler_y = scalers['y']
        print("[+] System Ready.")

    def predict(self, line1, line2, target_epoch_str):
        """
        Takes TLE + Target Time -> Returns Corrected State (r, v)
        """
        if self.scaler_X is None: self.load_artifacts()
        
        # 1. SGP4 Propagation (Baseline)
        sat = Satrec.twoline2rv(line1, line2, WGS72)
        
        # Calculate time difference in minutes
        jd_full = sat.jdsatepoch + sat.jdsatepochF
        ts_start = pd.Timestamp(jd_full - 2440587.5, unit='D')
        ts_target = pd.to_datetime(target_epoch_str)
        dt_minutes = (ts_target - ts_start).total_seconds() / 60.0
        
        e, r, v = sat.sgp4_tsince(dt_minutes)
        if e != 0: 
            return None, None # SGP4 Error
            
        # 2. Prepare PINN Input
        # Feature vector must match training columns exactly
        raw_features = np.array([
            r[0], r[1], r[2], v[0], v[1], v[2], 
            sat.bstar, sat.ndot, dt_minutes
        ]).reshape(1, -1)
        
        # 3. Scale & Tensorize
        X_scaled = self.scaler_X.transform(raw_features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(CONFIG['DEVICE'])
        
        # 4. Time Gate Input (Normalized 0-1 approx)
        t_norm = dt_minutes / 1440.0
        t_tensor = torch.tensor([[t_norm]], dtype=torch.float32).to(CONFIG['DEVICE'])
        
        # 5. Inference
        with torch.no_grad():
            pred_scaled = self.model(X_tensor, t_tensor).cpu().numpy()
            
        # 6. Inverse Scale Correction
        correction = self.scaler_y.inverse_transform(pred_scaled)[0]
        
        # 7. Apply Correction
        r_corr = np.array(r) + correction[:3]
        v_corr = np.array(v) + correction[3:]
        
        return np.array(r), r_corr

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    deployer = OrbitDeployer()
    
    # Check if we need to train
    if not os.path.exists(CONFIG['MODEL_PATH']):
        if os.path.exists(CONFIG['DATA_PATH']):
            deployer.train(CONFIG['DATA_PATH'])
        else:
            print(f"[!] Error: {CONFIG['DATA_PATH']} not found.")
            exit()
    else:
        deployer.load_artifacts()

    # --- SCENARIO: Starlink-3321 vs Debris ---
    print("\n" + "="*40)
    print("   CONJUNCTION ASSESSMENT SYSTEM")
    print("="*40)
    
    tle1_A = "1 50803U 22001A   22011.83334491  .00918374  26886-3 -20449-2 0  9990"
    tle2_A = "2 50803  53.2176 175.5863 0053823 179.7175 211.9048 15.94459142  2073"
    
    tle1_B = "1 99999U 22001A   22011.83334491  .00918374  26886-3 -20449-2 0  9990"
    tle2_B = "2 99999  53.2100 175.6000 0053823 179.7175 211.9048 15.94459142 42073"
    
    target_time = "2022-01-12 20:00:00"

    # Run Prediction
    sgp4_A, pinn_A = deployer.predict(tle1_A, tle2_A, target_time)
    sgp4_B, pinn_B = deployer.predict(tle1_B, tle2_B, target_time)

    # Calculate Metrics
    miss_sgp4 = np.linalg.norm(sgp4_A - sgp4_B)
    miss_pinn = np.linalg.norm(pinn_A - pinn_B)
    delta = miss_sgp4 - miss_pinn # Positive = PINN predicts closer approach

    print(f"Target Time: {target_time}")
    print(f"-"*30)
    print(f"Object A (Starlink):")
    print(f"  SGP4 Pos: {sgp4_A.astype(int)} km")
    print(f"  PINN Pos: {pinn_A.astype(int)} km")
    print(f"-"*30)
    print(f"Object B (Debris):")
    print(f"  SGP4 Pos: {sgp4_B.astype(int)} km")
    print(f"  PINN Pos: {pinn_B.astype(int)} km")
    print(f"-"*30)
    print(f"Miss Distance Analysis:")
    print(f"  SGP4 Miss Distance: {miss_sgp4:.4f} km")
    print(f"  PINN Miss Distance: {miss_pinn:.4f} km")
    
    if delta > 0:
        print(f"\n[ALERT] PINN predicts HIGHER RISK.")
        print(f"        Objects are {delta:.4f} km CLOSER than SGP4 predicts.")
    else:
        print(f"\n[OK] PINN predicts LOWER RISK.")
        print(f"     Objects are {abs(delta):.4f} km FURTHER APART.")
    print("="*40)