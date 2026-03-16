import json, os

cfg = os.path.abspath("config/models.json")

good = [
    {
        "id": "can_detector_v1",
        "display": "Can Detector V1",
        "enabled": True,
        "requires_gpu": False,
        "weights_path": "C:\\Code\\CAN_IDS_PROJECT\\models\\can_detector_v1\\weights.pth",
        "notes": "dummy placeholder - replace with real weights when available"
    }
]

print("Rewriting:", cfg)

os.makedirs(os.path.dirname(cfg), exist_ok=True)
with open(cfg, "w", encoding="utf-8") as f:
    json.dump(good, f, indent=2)

print("DONE. Preview:")
print(open(cfg,"r",encoding="utf-8").read())
