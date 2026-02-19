import torch
import numpy as np
import pandas as pd
import joblib
import os
from sgp4.api import Satrec, WGS72
from src.model import GatedPINN
from src.utils import set_seed

class OrbitDeployer:
    def __init__(self, model_path='models/pinn_model.pth', scaler_x_path='models/scaler_X.pkl', scaler_y_path='models/scaler_Y.pkl'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GatedPINN().to(self.device)
        self.model_path = model_path
        self.scaler_x_path = scaler_x_path
        self.scaler_y_path = scaler_y_path
        self.scaler_X = None
        self.scaler_y = None
        self.load_artifacts()

    def load_artifacts(self):
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            # Handle if state_dict is nested under 'model_state_dict' or similar
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # print(f"DEBUG: Loaded keys: {list(state_dict.keys())[:5]}")
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[!] Warning: Missing keys: {missing[:5]}...")
            if unexpected:
                print(f"[!] Warning: Unexpected keys: {unexpected[:5]}...")
                
        except Exception as e:
            print(f"[!] Critical Error loading model: {e}")
            raise e
        
        self.model.eval()
        
        # print("[-] Loading scalers...")
        self.scaler_X = joblib.load(self.scaler_x_path)
        self.scaler_y = joblib.load(self.scaler_y_path)
        # print("[+] System Ready.")

    def predict(self, line1, line2, target_epoch_str):
        """
        Returns:
            initial_state: (r0, v0) at TLE epoch
            sgp4_state: (r, v) at target time
            pinn_state: (r_corr, v_corr) at target time
        """
        # 1. SGP4 Propagation (Baseline)
        sat = Satrec.twoline2rv(line1, line2, WGS72)
        
        # Calculate time difference in minutes
        jd_full = sat.jdsatepoch + sat.jdsatepochF
        ts_start = pd.Timestamp(jd_full - 2440587.5, unit='D')
        ts_target = pd.to_datetime(target_epoch_str)
        dt_minutes = (ts_target - ts_start).total_seconds() / 60.0
        
        # Get Initial State (t=0)
        e0, r0, v0 = sat.sgp4_tsince(0)
        
        # Get Final State (SGP4)
        e, r, v = sat.sgp4_tsince(dt_minutes)
        
        if e != 0 or e0 != 0: 
            return None, None, None # SGP4 Error
            
        # 2. Prepare PINN Input
        raw_features = np.array([
            r[0], r[1], r[2], v[0], v[1], v[2], 
            sat.bstar, sat.ndot, dt_minutes
        ]).reshape(1, -1)
        
        # 3. Scale & Tensorize
        X_scaled = self.scaler_X.transform(raw_features)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # 4. Time Gate
        t_norm = dt_minutes / 1440.0
        t_tensor = torch.tensor([[t_norm]], dtype=torch.float32).to(self.device)
        
        # 5. Inference
        with torch.no_grad():
            pred_scaled = self.model(X_tensor, t_tensor).cpu().numpy()
            
        # 6. Inverse Scale Correction
        correction = self.scaler_y.inverse_transform(pred_scaled)[0]
        
        # 7. Apply Correction
        r_corr = np.array(r) + correction[:3]
        v_corr = np.array(v) + correction[3:]
        
        return (np.array(r0), np.array(v0)), (np.array(r), np.array(v)), (r_corr, v_corr)

    def get_trajectory(self, line1, line2, target_epoch_str, steps=300):
        """
        Generates SGP4 trajectory points for a full orbital window around the target.
        Range: T-50 mins to T+50 mins to visualize the full 'oval' or arc.
        """
        sat = Satrec.twoline2rv(line1, line2, WGS72)
        jd_full = sat.jdsatepoch + sat.jdsatepochF
        ts_start = pd.Timestamp(jd_full - 2440587.5, unit='D')
        ts_target = pd.to_datetime(target_epoch_str)
        
        # Minutes from TLE epoch to Target
        minutes_to_target = (ts_target - ts_start).total_seconds() / 60.0
        
        # Window: +/- 50 minutes around target (approx 1 full LEO orbit ~90-100 mins)
        t_start = minutes_to_target - 50
        t_end = minutes_to_target + 50
        
        times = np.linspace(t_start, t_end, steps)
        positions = []
        
        for t in times:
            e, r, v = sat.sgp4_tsince(t)
            if e == 0:
                positions.append(r)
                
        return np.array(positions)
