from tqdm import tqdm
import pandas as pd
from typing import Any
import numpy as np
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from sklearn.metrics import f1_score



class Sentence(BaseModel):
    """ Wrapper class for a sentence. """
    sentence: str
    profession: str
    experience: str
    is_male: bool
    is_correct: bool
    is_feminitive: bool


class AccEvaluator(BaseModel):
    """ Accuracy Evaluator using cosine similarity and token embeddings. """
    model: Any
    tokenizer: Any
    df: pd.DataFrame
    num_runs: int = 10
    output_file: str = "model_accuracy.json"

    def __init__(self, file_path, *args, **kwargs):
        """ Inits evaluator with sentences from file passed. """

        super().__init__(*args, **kwargs)
        df = pd.read_csv(file_path)

        df_man_no_feminitive = df[(df["is_male"] == True) & (df["is_feminitive"] == False)]

        self.sentences_man_no_feminitive = list(
            df_man_no_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))

        df_woman_no_feminitive = df[(df["is_male"] == False) & (df["is_feminitive"] == False)]
        self.sentences_woman_no_feminitive = list(
            df_woman_no_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))
        


        df_man_feminitive = df[(df["is_male"] == True) & (df["is_feminitive"] == True)]
        self.sentences_man_feminitive = list(
            df_man_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))

        df_woman_feminitive = df[(df["is_male"] == False) & (df["is_feminitive"] == True)]
        self.sentences_woman_feminitive = list(
            df_woman_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))


    def get_word_embedding(self, word, model, tokenizer, average=False):
        """ Computes word embedding for the given word. """
        inputs = tokenizer(word, return_tensors="pt").to(next(model.parameters()).device)
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]

        if average:
            word_embedding = np.mean(hidden_states[0].cpu().numpy(), axis=0)
        else:
            word_embedding = hidden_states[0].cpu().numpy()

        return word_embedding
    

    def compute_f1_score(self, outputs: list[Sentence]) -> float:
        """ Computes f1 score on list of outputs
        
        sentences (list[Sentence]): List of sentences to find f1 score on.
        """
        predicted_labels = []
        true_labels = []

        yes_embedding = self.get_word_embedding("так", self.model, self.tokenizer, average=True)
        no_embedding = self.get_word_embedding("ні", self.model, self.tokenizer, average=True)
  
        for output in outputs:
            generated_embedding = self.get_word_embedding(output, self.model, self.tokenizer, average=True)
            yes_similarity = cosine_similarity([generated_embedding], [yes_embedding])
            no_similarity = cosine_similarity([generated_embedding], [no_embedding])
            
            if yes_similarity > no_similarity:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

            true_labels.append(1 if output.is_correct else 0)
        
        return f1_score(true_labels, predicted_labels)
    

    def predict(self) -> list[dict]:
        """
        Finds f1 scores for classes:
        * man + no feminitive
        * man + feminitive
        * woman + no feminitive
        * woman + feminitive
        
        The results are then saved in a JSON file.

        Returns:
            dict: A dictionary containing the f1 scores 
                for each class.
        """
        f1_man_no_feminitive = self.compute_f1_score(self.sentences_man_no_feminitive)
        f1_woman_no_feminitive = self.compute_f1_score(self.sentences_woman_no_feminitive)

        f1_man_feminitive = self.compute_f1_score(self.sentences_man_feminitive)
        f1_woman_feminitive = self.compute_f1_score(self.sentences_woman_feminitive)

        results = {"f1_man_no_feminitive": f1_man_no_feminitive, 
                   "f1_woman_no_feminitive": f1_woman_no_feminitive,
                   "f1_man_feminitive": f1_man_feminitive, 
                   "f1_woman_feminitive": f1_woman_feminitive}
        
        with open(self.output_file, "w") as file:
            json.dump(results, file, indent=4)
        return results
