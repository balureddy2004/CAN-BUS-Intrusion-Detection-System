# scripts/test_run_training.py
from backend.train_job import run_training, DummyEvent
import time
import os

def broadcast(ev):
    print("BCAST:", ev)

def record(ev):
    print("RECORD:", ev)

def main():
    # Use a safe local model output directory to avoid permission issues
    model_out_dir = "tmp_models"
    os.makedirs(model_out_dir, exist_ok=True)

    params = {
        "model": "RandomForest",
        "epochs": 3,
        "batchSize": 2,
        "dataset_path": "backend/test_small.csv",
        "maxRows": 1000,
        "trainSplit": 0.8,
        "seed": 42,
    }
    stop_ev = DummyEvent()
    pause_ev = DummyEvent()
    state = {}
    print("STARTING TRAINING (demo)...")
    run_training(params,
                 broadcast_train=broadcast,
                 record_fn=record,
                 stop_event=stop_ev,
                 pause_event=pause_ev,
                 state=state,
                 model_out_dir=model_out_dir)
    print("FINAL STATE:", state)

if __name__ == "__main__":
    main()
