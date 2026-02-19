import torch
import dash_mantine_components as dmc
import os

model_path1 = "models/pinn_model.pth" # This is currently a copy of best_pinn_model.pth
model_path2 = "notebooks_temp/orbit_error_model.pth"

print(f"DMC Version: {dmc.__version__}")

def check_model(path):
    print(f"\nChecking {path}...")
    if not os.path.exists(path):
        print("File not found.")
        return
    try:
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict):
            print("Keys:", list(state.keys())[:5])
            if 'net.0.weight' in state:
                print("Model weights found directly.")
            elif 'model_state_dict' in state:
                print("Found 'model_state_dict' key.")
                print("Inner Keys:", list(state['model_state_dict'].keys())[:5])
            else:
                print("Unknown dict structure.")
        else:
            print("Not a dict.")
    except Exception as e:
        print(f"Error: {e}")

check_model(model_path1)
check_model(model_path2)
