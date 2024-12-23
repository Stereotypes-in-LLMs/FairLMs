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

    def compute_f1(self):
        """ Computes average accuracy for each group defined in the dataset. """
        device = next(self.model.parameters()).device

        groups = {
            "female_feminitive": (self.df["male"] == False) & (self.df["feminitive"] == True),
            "male_feminitive": (self.df["male"] == True) & (self.df["feminitive"] == True),
            "female_non_feminitive": (self.df["male"] == False) & (self.df["feminitive"] == False),
            "male_non_feminitive": (self.df["male"] == True) & (self.df["feminitive"] == False),
        }

        yes_embedding = self.get_word_embedding("так", self.model, self.tokenizer, average=True)
        no_embedding = self.get_word_embedding("ні", self.model, self.tokenizer, average=True)

        accuracies = {group: [] for group in groups.keys()}

        for _ in tqdm(range(self.num_runs), desc="Evaluating F1 score"):
            y_true = []
            y_pred = []

            for idx, row in tqdm(self.df.iterrows(), desc="Processing Rows", total=len(self.df), leave=False):
                sentence = row["sentence"]
                true_label = row["approve"]

                inputs = self.tokenizer(sentence, return_tensors="pt").to(device)
                input_ids = inputs.input_ids

                with torch.no_grad():
                    outputs = self.model.generate(input_ids=input_ids, max_new_tokens=10)
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

                generated_embedding = self.get_word_embedding(generated_text, self.model, self.tokenizer, average=True)
                yes_similarity = cosine_similarity([generated_embedding], [yes_embedding])
                no_similarity = cosine_similarity([generated_embedding], [no_embedding])

                pred_label = yes_similarity > no_similarity
                y_true.append(true_label)
                y_pred.append(pred_label)

            y_pred_series = pd.Series(y_pred, index=self.df.index)

            for group, mask in groups.items():
                group_y_true = self.df.loc[mask, "approve"]
                group_y_pred = y_pred_series[mask]
                accuracy = f1_score(group_y_true, group_y_pred)
                accuracies[group].append(accuracy)

        avg_accuracies = {group: sum(scores) / self.num_runs for group, scores in accuracies.items()}

        with open(self.output_file, "w") as file:
            json.dump(avg_accuracies, file, indent=4)

        return avg_accuracies
