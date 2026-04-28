import torch
from train import is_bitnet_model, apply_lora, detect_target_modules
from config import BASE_MODEL_NAME, BITNET_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockModule:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return "Linear"

class MockModel:
    def __init__(self):
        self.modules_dict = {
            "model.layers.0.self_attn.q_proj": MockModule("q_proj"),
            "model.layers.0.self_attn.k_proj": MockModule("k_proj"),
            "model.layers.0.self_attn.v_proj": MockModule("v_proj"),
            "model.layers.0.mlp.gate_proj": MockModule("gate_proj"),
        }
    def named_modules(self):
        return self.modules_dict.items()

def test_logic():
    print(f"Testing with model: {BASE_MODEL_NAME}")
    is_bitnet = is_bitnet_model(BASE_MODEL_NAME)
    print(f"Is BitNet: {is_bitnet}")

    mock_model = MockModel()
    modules = detect_target_modules(mock_model)
    print(f"Detected modules: {modules}")

    # We can't easily call apply_lora without a real model object from Unsloth
    # but we can verify the conditional logic we added
    lora_r = BITNET_CONFIG['lora_r'] if is_bitnet else 32
    print(f"Expected LoRA R: {lora_r}")

if __name__ == "__main__":
    test_logic()
