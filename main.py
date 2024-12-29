import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from parser.parser import parse_config
from debias.soft_debias import SoftDebiasModelWrapper
from metrics.prob_metric import ProbEvaluator

def load_model(model_path, adapter_path):
    """
    Load model and tokenizer
    """
    base_model = model_path
    adapter_model = adapter_path

    model = AutoModelForCausalLM.from_pretrained(base_model, load_in_8bit=True)
    model = PeftModel.from_pretrained(model, adapter_model, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.eval()
    return model, tokenizer

def main(config_path):
    params = parse_config(config_path=config_path)
    model_orig, tokenizer = load_model(params["model_path"], params["adapter_path"])
    if params["debias"] == "soft":
        model = SoftDebiasModelWrapper(model = model_orig, tokenizer = tokenizer,
                                    gender_defining_man = params["male_words_ukr"], gender_defining_woman = params["female_words_ukr"], 
                                    pca_component_s = params["pca_s"], pca_component_e = params["pca_e"])
        print("here")
    elif params["debias"] == "":
        # No debias
        model = model_orig
    else:
        print(f"No such debias methods like {params['debias']}")

    evaluator = ProbEvaluator(file_path = params["dataset_path"], tokenizer = tokenizer, model = model)
    predictions = evaluator.predict()

    print(predictions)

if __name__ == "__main__":
    print("Test started")
    main("/path/to/config")