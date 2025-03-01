from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from parser.parser import parse_config
from debias.soft_debias import SoftDebiasModelWrapper
from pydantic import BaseModel
from typing import Any
from debias import SoftDebiasModelWrapper, PrompDebiasModelWrapper, PlainModelWrapper
from metrics import AccEvaluator, DiffEvaluator
from sentence_transformers import SentenceTransformer
import json


def load_model(model_path):
    """
    Loads model and tokenizer
    """

    if model_path == "openlm-research/open_llama_7b_v2":
        model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b_v2", load_in_8bit=True)
        model = PeftModel.from_pretrained(model, "robinhad/open_llama_7b_uk", load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b_v2")
        model.eval()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            load_in_8bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  load_in_8bit=True)
    return model, tokenizer


class Runner(BaseModel):
    model_orig: Any = None
    model: Any = None 
    tokenizer: Any = None 

    def run(self, params: dict = None, config_path: str = None, preload_model: bool = True, result_file_path = None):
        if not params and not config_path:
            raise ValueError("Error: Provide params(dict) or path to config file confing_path(str).")
        if config_path:
            params = parse_config(config_path=config_path)
    
        if preload_model:
            model_orig, tokenizer = load_model(params["model_path"])
            self.model_orig = model_orig
            self.tokenizer = tokenizer

        else:
            model_orig = self.model_orig
            tokenizer = self.tokenizer 
            print(tokenizer)
    
        if params["debias"] == "soft":
            model = SoftDebiasModelWrapper(model_name = params["model_path"], model = model_orig, tokenizer = tokenizer,
                                        gender_defining_man = params["male_words_ukr"], gender_defining_woman = params["female_words_ukr"], 
                                        pca_component_s = params["pca_s"], pca_component_e = params["pca_e"], hard_like = False)
            self.model = model
        elif params["debias"] == "hard":
            model = SoftDebiasModelWrapper(model_name = params["model_path"], model = model_orig, tokenizer = tokenizer,
                                        gender_defining_man = params["male_words_ukr"], gender_defining_woman = params["female_words_ukr"], 
                                        pca_component_s = params["pca_s"], pca_component_e = params["pca_e"], hard_like = True)
            self.model = model
        elif params["debias"] == "prompt":
            model = PrompDebiasModelWrapper(model_name = params["model_path"], model = model_orig, tokenizer = tokenizer, debias_prompt = params["debias_promp"])
            self.model = model

        elif params["debias"] == "":
            model = PlainModelWrapper(model_name = params["model_path"], model = model_orig, tokenizer = tokenizer)
            self.model = model
        else:
            print(f"No such debias methods like {params['debias']}")
    
        result = {}
        if "acc" in params["metrics"]:
            embeddings_model = SentenceTransformer(params["embeddings_model"])
            evaluator = AccEvaluator(file_path = params["acc_dataset_path"], embeddings_model = embeddings_model,
                                      tokenizer = tokenizer, model = model, dataset_percentage = params["dataset_percentage"])
            predictions = evaluator.predict()
            result["acc"] = predictions
        if "diff" in params["metrics"]:
            evaluator = DiffEvaluator(file_path = params["prob_dataset_path"], tokenizer = tokenizer, model = model,
                                      dataset_percentage = params["dataset_percentage"])
            predictions = evaluator.predict()
            result["diff"] = predictions
        if result_file_path:
            with open(result_file_path, 'w') as f:
                json.dump(result, f, indent=4)
        return result

def main(config_path):
    params = parse_config(config_path=config_path)
    runner = Runner()
    runner.run(params, preload_model = False, result_file_path = 'data.json')

if __name__ == "__main__":
    print("Test started")
    main("/path/to/config")