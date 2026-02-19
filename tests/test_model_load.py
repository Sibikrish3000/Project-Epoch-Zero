import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.deployer import OrbitDeployer

def test_model_load():
    try:
        print("Testing OrbitDeployer initialization...")
        deployer = OrbitDeployer()
        print("OrbitDeployer initialized successfully.")
        
        # Test prediction with mock TLE
        tle1 = "1 50803U 22001A   22011.83334491  .00918374  26886-3 -20449-2 0  9990"
        tle2 = "2 50803  53.2176 175.5863 0053823 179.7175 211.9048 15.94459142  2073"
        target_time = "2022-01-12 20:00:00"
        
        print("Testing prediction...")
        r_sgp4, v_sgp4, r_pinn = deployer.predict(tle1, tle2, target_time)
        
        if r_sgp4 is not None and r_pinn is not None:
            print("Prediction successful.")
            print(f"SGP4 Position: {r_sgp4}")
            print(f"PINN Position: {r_pinn}")
        else:
            print("Prediction failed (returned None).")
            exit(1)
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        exit(1)

if __name__ == "__main__":
    test_model_load()
