from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd
from typing import Any
import numpy as np
import json
import torch
from sklearn.metrics import f1_score

class Sentence(BaseModel):
    """ Wrapper class for a sentence. """
    sentence: str
    profession: str
    experience: str
    is_male: bool
    is_correct: bool
    is_feminitive: bool


class ProbEvaluator(BaseModel):
    """ Probability Evaluator """
    tokenizer: Any
    model: Any
    sentences_man_no_feminitive: list[Sentence] = None
    sentences_woman_no_feminitive: list[Sentence] = None

    sentences_man_feminitive: list[Sentence] = None
    sentences_woman_feminitive: list[Sentence] = None

    UNCONDITIONAL_START_TOKEN: str = "<s>"
    device: str = "cuda"
    output_file: str = "model_output.json"


    def compute_sentence_probability(self, sentence: str):
        """ Computes sentence probability
        sentence (str): Sentence to find probability for.
        """
        start_token = torch.tensor(self.tokenizer.encode(
            self.UNCONDITIONAL_START_TOKEN)).to(self.device).unsqueeze(0)
        initial_token_probabilities = self.model(start_token)

        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1)


        tokens = self.tokenizer.encode(sentence)

        joint_sentence_probability = [
            initial_token_probabilities[0, 0, tokens[0]].item()]

        tokens_tensor = torch.tensor(
            tokens).to(self.device).unsqueeze(0)

        output = torch.softmax(self.model(tokens_tensor)[0], dim=-1)
        for idx in range(1, len(tokens)):
            joint_sentence_probability.append(
                output[0, idx-1, tokens[idx]].item())

        assert len(tokens) == len(joint_sentence_probability)

        score = np.sum([np.log2(i)
                        for i in joint_sentence_probability])
        score /= len(joint_sentence_probability)
        score = np.power(2, score)

        return score


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


    def compute_f1_score(self, sentences: list[Sentence]) -> float:
        """ Computes f1 score on list of sentences
        
        sentences (list[Sentence]): List of sentences to find f1 score on.
        """
        predicted_labels = []
        true_labels = []

        
        for sentence in sentences:
            prob_has_position = self.compute_sentence_probability(sentence.sentence.replace("BLANK ", ""))
            prob_has_no_position = self.compute_sentence_probability(sentence.sentence.replace("BLANK ", "не "))
            
            if prob_has_position > prob_has_no_position:
                predicted_labels.append(1)
            elif prob_has_position < prob_has_no_position:
                predicted_labels.append(0)

            true_labels.append(1 if sentence.is_correct else 0)
        
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
    
