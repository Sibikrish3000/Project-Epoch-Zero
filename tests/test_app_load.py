import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import app

def test_app_load():
    try:
        print("Testing App initialization...")
        assert app is not None
        assert app.layout is not None
        print("App initialized successfully.")
    except Exception as e:
        print(f"App load failed: {e}")
        exit(1)

if __name__ == "__main__":
    test_app_load()
