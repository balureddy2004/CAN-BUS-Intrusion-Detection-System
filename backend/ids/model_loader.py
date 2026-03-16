from core.logger import get_logger

log = get_logger(__name__)

# Placeholder model – replace with real loader (joblib/torch/onnx/etc.)
class DummyModel:
    def predict_proba(self, features):
        # return (p_benign, p_attack)
        return 0.97, 0.03

MODEL = DummyModel()

def load_model():
    global MODEL
    log.info("Loaded model: dummy")
    return MODEL
