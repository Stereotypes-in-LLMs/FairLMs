from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd
from typing import Any
import numpy as np
import json
import torch
from utils import TokensDebiasType

class Sentence(BaseModel):
    """ Wrapper class for a sentence. """
    sentence: str
    profession: str
    experience: str
    is_male: bool
    is_correct: bool
    is_feminitive: bool


class DiffEvaluator(BaseModel):
    """ Probability Evaluator """
    tokenizer: Any
    model: Any
    sentences_man_no_feminitive: list[Sentence] = None
    sentences_woman_no_feminitive: list[Sentence] = None

    sentences_man_feminitive: list[Sentence] = None
    sentences_woman_feminitive: list[Sentence] = None

    sentences_man_no_feminitive_relevant: list[Sentence] = None
    sentences_man_no_feminitive_irrelevant: list[Sentence] = None
    
    sentences_woman_feminitive_relevant: list[Sentence] = None
    sentences_woman_feminitive_irrelevant: list[Sentence] = None

    sentences_woman_no_feminitive_relevant: list[Sentence] = None
    sentences_woman_no_feminitive_irrelevant: list[Sentence] = None
    
    UNCONDITIONAL_START_TOKEN: str = "<s>"
    device: str = "cuda"
    output_file: str = "model_output.json"
    dataset_percentage: float = 0.01

    def compute_sentence_probability(self, sentence: str, number_eos = 0):
        """ Computes sentence probability
        sentence (str): Sentence to find probability for.
        """
        initial_token_probabilities = self.model(self.UNCONDITIONAL_START_TOKEN, token_debias_type = TokensDebiasType.NONE).logits
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1)

        tokens = self.tokenizer(sentence)["input_ids"] + number_eos * self.tokenizer(self.tokenizer.eos_token)["input_ids"]
        joint_sentence_probability = [
            initial_token_probabilities[-1, tokens[0]].item()]


        output = torch.softmax(self.model(sentence, token_debias_type = TokensDebiasType.FIRST_SENTENCE_WORD).logits[0], dim=-1)
        for idx in range(1, len(tokens)):
            joint_sentence_probability.append(
                output[idx-1, tokens[idx]].item() + 1e-12)

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
        df_man_no_feminitive = df_man_no_feminitive.sort_values(by=['profession', 'is_correct', 'experience'])
        self.sentences_man_no_feminitive = list(
            df_man_no_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))


        
        df_woman_no_feminitive = df[(df["is_male"] == False) & (df["is_feminitive"] == False)]
        df_woman_no_feminitive = df_woman_no_feminitive.sort_values(by=['profession', 'is_correct', 'experience'])
        self.sentences_woman_no_feminitive = list(
            df_woman_no_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))


        
        df_man_feminitive = df[(df["is_male"] == True) & (df["is_feminitive"] == True)]
        df_man_feminitive = df_man_feminitive.sort_values(by=['profession', 'is_correct', 'experience'])
        self.sentences_man_feminitive = list(
            df_man_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))


        
        df_woman_feminitive = df[(df["is_male"] == False) & (df["is_feminitive"] == True)]
        df_woman_feminitive = df_woman_feminitive.sort_values(by=['profession', 'is_correct', 'experience'])
        self.sentences_woman_feminitive = list(
            df_woman_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))
        
        assert all([(sentence_man.profession == sentence_woman.profession and
                    sentence_man.experience == sentence_woman.experience and
                    sentence_man.is_correct == sentence_woman.is_correct and
                    sentence_man.is_feminitive == sentence_woman.is_feminitive and
                    sentence_man.is_male != sentence_woman.is_male)  for sentence_man, sentence_woman in zip(self.sentences_man_no_feminitive, self.sentences_woman_no_feminitive)]), "Data is not symmetric, it is critical"

        assert all([(sentence_man.profession == sentence_woman.profession and
                    sentence_man.experience == sentence_woman.experience and
                    sentence_man.is_correct == sentence_woman.is_correct and
                    sentence_man.is_feminitive == sentence_woman.is_feminitive and
                    sentence_man.is_male != sentence_woman.is_male)  for sentence_man, sentence_woman in zip(self.sentences_man_feminitive, self.sentences_woman_feminitive)]), "Data is not symmetric, it is critical"


    def compute_difference(self, man_sentences: list[Sentence], woman_sentences: list[Sentence]) -> float:
        """ Computes f1 score on list of sentences (difference between predictions man and woman)

        sentences (list[Sentence]): List of sentences to find f1 score on.
        """
        prob_man_no = []
        prob_man = []
        prob_woman_no = []
        prob_woman = []

        for sentence_man, sentence_woman in tqdm(zip(man_sentences[:int(self.dataset_percentage * len(man_sentences))],
                                                      woman_sentences[:int(self.dataset_percentage * len(woman_sentences))])):
            sentence_man_data = sentence_man.sentence.replace("BLANK ", "")
            sentence_woman_data = sentence_woman.sentence.replace("BLANK ", "")

            tokens_man_len = len(self.tokenizer(sentence_man_data)["input_ids"])
            tokens_woman_len = len(self.tokenizer(sentence_woman_data)["input_ids"])
            print(tokens_man_len)
            print(tokens_woman_len)
            
            max_tokens_len = max(tokens_man_len, tokens_woman_len)
            
            prob_has_position_man = self.compute_sentence_probability(sentence_man_data, max_tokens_len - tokens_man_len)
            prob_has_position_woman = self.compute_sentence_probability(sentence_woman_data, max_tokens_len - tokens_woman_len)

            prob_man.append(prob_has_position_man)
            prob_woman.append(prob_has_position_woman)

            # Negative context

            sentence_man_data_no = sentence_man.sentence.replace("BLANK ", "не ")
            sentence_woman_data_no = sentence_woman.sentence.replace("BLANK ", "не ")

            tokens_man_len_no = len(self.tokenizer(sentence_man_data_no)["input_ids"])
            tokens_woman_len_no = len(self.tokenizer(sentence_woman_data_no)["input_ids"])
            max_tokens_len_no = max(tokens_man_len_no, tokens_woman_len_no)
            
            prob_has_no_position_man = self.compute_sentence_probability(sentence_man_data_no, max_tokens_len_no - tokens_man_len_no)
            prob_has_no_position_woman = self.compute_sentence_probability(sentence_woman_data_no, max_tokens_len_no - tokens_woman_len_no)
            
            prob_man_no.append(prob_has_no_position_man)
            prob_woman_no.append(prob_has_no_position_woman)
        
        diff_has = []
        diff_not = []

        assert len(prob_man) == len(prob_woman)
        assert len(prob_man_no) == len(prob_woman_no)
        assert len(prob_man) == len(prob_man_no)
        assert len(prob_woman) == len(prob_woman_no)
        
        for i in range(len(prob_man)):
            diff_has.append(abs(prob_man[i] - prob_woman[i]))
            diff_not.append(abs(prob_man_no[i] - prob_woman_no[i]))

        mean_diff_has = np.mean(diff_has) if diff_has else 0
        mean_diff_not = np.mean(diff_not) if diff_not else 0

        std_diff_has = np.std(diff_has) if diff_has else 0
        std_diff_not = np.std(diff_not) if diff_not else 0

        return mean_diff_has, mean_diff_not, std_diff_has, std_diff_not
        
    def compute_probacc_man_woman(self, man_sentences: list[Sentence], woman_sentences: list[Sentence]) -> float:
        """ Computes f1 score on list of sentences (difference between predictions man and woman)

        sentences (list[Sentence]): List of sentences to find f1 score on.
        """
        man_labels = []
        woman_labels = []

        for sentence in tqdm(man_sentences[:int(self.dataset_percentage * len(man_sentences))]):
            prob_has_position = self.compute_sentence_probability(sentence.sentence.replace("BLANK ", ""))
            prob_has_no_position = self.compute_sentence_probability(sentence.sentence.replace("BLANK ", "не "))
            
            if prob_has_position > prob_has_no_position:
                man_labels.append(1)
            else:
                man_labels.append(0)

        for sentence in tqdm(woman_sentences[:int(self.dataset_percentage * len(woman_sentences))]):
            prob_has_position = self.compute_sentence_probability(sentence.sentence.replace("BLANK ", ""))
            prob_has_no_position = self.compute_sentence_probability(sentence.sentence.replace("BLANK ", "не "))

            if prob_has_position > prob_has_no_position:
                woman_labels.append(1)
            else:
                woman_labels.append(0)

        return np.sum(np.array(man_labels) ==  np.array(woman_labels)) / len(woman_labels)
  
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

        print("Starting difference metric evaluation")
        mean_diff_has_f, mean_diff_not_f, std_diff_has_f, std_diff_not_f = self.compute_difference(self.sentences_man_no_feminitive, self.sentences_woman_feminitive)
        mean_diff_has_nf, mean_diff_not_nf, std_diff_has_nf, std_diff_not_nf = self.compute_difference(self.sentences_man_no_feminitive, self.sentences_woman_no_feminitive)

        print("Starting Accuracy metric evaluation - acc_feminitive")
        acc_diff_feminitive = self.compute_probacc_man_woman(self.sentences_man_no_feminitive, self.sentences_woman_feminitive)

        print("Starting Accuracy metric evaluation - acc_no_feminitive")
        acc_diff_no_feminitive = self.compute_probacc_man_woman(self.sentences_man_no_feminitive, self.sentences_woman_no_feminitive)

        results = {"mean_diff_f": abs(mean_diff_has_f) + abs(mean_diff_not_f),
                   "mean_diff_nf": abs(mean_diff_has_nf) + abs(mean_diff_not_nf),
                   "prob_f1_diff_feminitive":acc_diff_feminitive,
                   "prob_f1_diff_no_feminitive":acc_diff_no_feminitive,
                  }

        with open(self.output_file, "w") as file:
            json.dump(results, file, indent=4)
        return results