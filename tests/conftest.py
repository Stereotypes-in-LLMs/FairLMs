import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B-Instruct"]


def tiny_model(model_id):
    """Random-weight 2-layer model with the real architecture, vocabulary and tokenizer (CPU, float32)."""
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:2]
    if "gemma" in model_id:
        cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    return model, AutoTokenizer.from_pretrained(model_id)


@pytest.fixture(scope="session", params=MODELS, ids=lambda m: m.split("/")[-1])
def tiny(request):
    try:
        return tiny_model(request.param)
    except OSError as e:  # tokenizer/config not cached and no access
        pytest.skip(f"{request.param} unavailable: {e}")
