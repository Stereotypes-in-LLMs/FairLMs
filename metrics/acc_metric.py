from tqdm import tqdm
import pandas as pd
from typing import Any
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import f1_score

from enum import Enum

class TokensDebiasType(Enum):
    LAST_WORD = "last_word"
    FIRST_SENTENCE_WORD = "first_sentence_word"
    NONE = "none"


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
    embeddings_model: Any
    tokenizer: Any
    num_runs: int = 10
    output_file: str = "model_accuracy.json"
    sentences_man_no_feminitive: list[Sentence] = None
    sentences_woman_no_feminitive: list[Sentence] = None

    sentences_man_feminitive: list[Sentence] = None
    sentences_woman_feminitive: list[Sentence] = None
    device: str = "cuda"
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset_percentage: float = 0.01

    def __init__(self, file_path, *args, **kwargs):
        """ Inits evaluator with sentences from file passed. """

        super().__init__(*args, **kwargs)
        df = pd.read_csv(file_path)

        
        df_man_no_feminitive = df[(df["is_male"] == True) & (df["is_feminitive"] == False)]
        df_man_no_feminitive = df_man_no_feminitive.sort_values(by=['profession', 'experience'])
        self.sentences_man_no_feminitive = list(
            df_man_no_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))


        
        df_woman_no_feminitive = df[(df["is_male"] == False) & (df["is_feminitive"] == False)]
        df_woman_no_feminitive = df_woman_no_feminitive.sort_values(by=['profession', 'experience'])
        self.sentences_woman_no_feminitive = list(
            df_woman_no_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))


        
        df_man_feminitive = df[(df["is_male"] == True) & (df["is_feminitive"] == True)]
        df_man_feminitive = df_man_feminitive.sort_values(by=['profession', 'experience'])
        self.sentences_man_feminitive = list(
            df_man_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))


        
        df_woman_feminitive = df[(df["is_male"] == False) & (df["is_feminitive"] == True)]
        df_woman_feminitive = df_woman_feminitive.sort_values(by=['profession', 'experience'])
        self.sentences_woman_feminitive = list(
            df_woman_feminitive.apply(lambda row: Sentence(**row.to_dict()), axis=1))

        assert len(self.sentences_man_no_feminitive) == len(self.sentences_woman_feminitive)
        assert len(self.sentences_man_no_feminitive) == len(self.sentences_woman_no_feminitive)
        

        assert all([(sentence_man.profession == sentence_woman.profession and
                    sentence_man.experience == sentence_woman.experience and
                    sentence_man.is_correct == sentence_woman.is_correct and
                    sentence_man.is_feminitive == sentence_woman.is_feminitive and
                    sentence_man.is_male != sentence_woman.is_male)  for sentence_man, sentence_woman in zip(self.sentences_man_no_feminitive, self.sentences_woman_no_feminitive)]), "Data is not symmetric, it is critical"

        assert all([(sentence_man.profession == sentence_woman.profession and
                    sentence_man.experience == sentence_woman.experience and
                    sentence_man.is_correct == sentence_woman.is_correct and
                    sentence_man.is_feminitive != sentence_woman.is_feminitive and
                    sentence_man.is_male != sentence_woman.is_male)  for sentence_man, sentence_woman in zip(self.sentences_man_no_feminitive, self.sentences_woman_feminitive)]), "Data is not symmetric, it is critical"        
    
    def get_sentence_embedding(self, sentence: str, embeddings_model):
        """ Computes word embedding for the given sentence. """
        embeddings = embeddings_model.encode([sentence.lower()], show_progress_bar=False)
        return embeddings

    def generate_sentence(self, sentence: str, model, tokenizer, num_tokens = 5):
        for _ in range(num_tokens):
            output = model(sentence, token_debias_type = TokensDebiasType.LAST_WORD, )
            # Decode the generated tokens
            generated_text = tokenizer.decode(output.logits[0][-1].argmax(dim=-1), skip_special_tokens=True)
            # Append the result to the list
            sentence += generated_text
        return sentence

    def compute_f1_score(self, sentences: list[Sentence]) -> float:
        """ Computes f1 score on list of outputs

        sentences (list[Sentence]): List of sentences to find f1 score on.
        """
        predicted_labels = []
        true_labels = []

        yes_embedding = self.get_sentence_embedding("так", self.embeddings_model)
        no_embedding = self.get_sentence_embedding("ні", self.embeddings_model)

        for output in tqdm(sentences[:int(self.dataset_percentage * len(sentences))]):
            generated_response = self.generate_sentence("Відповідай так чи ні: " + output.sentence, self.model, self.tokenizer)
            generated_embedding = self.get_sentence_embedding(generated_response[len("Відповідай так чи ні: " + output.sentence):], self.embeddings_model)

            yes_similarity = cosine_similarity(generated_embedding, yes_embedding)[0][0]
            no_similarity = cosine_similarity(generated_embedding, no_embedding)[0][0]
            if yes_similarity > no_similarity:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

            true_labels.append(1 if output.is_correct else 0)

        return f1_score(true_labels, predicted_labels), predicted_labels

    def compute_f1_score_man_woman(self, sentences_man: list[Sentence], sentences_woman: list[Sentence]) -> float:
        """ Computes f1 score on list of sentences (difference between predictions man and woman)

        sentences (list[Sentence]): List of sentences to find f1 score on.
        """
        man_labels = []
        woman_labels = []

        yes_embedding = self.get_sentence_embedding("так", self.embeddings_model)
        no_embedding = self.get_sentence_embedding("ні", self.embeddings_model)

        for sentence_man in tqdm(sentences_man[:int(self.dataset_percentage * len(sentences_man))]):
          generated_response = self.generate_sentence("Відповідай так чи ні: " + sentence_man.sentence, self.model, self.tokenizer)
          generated_embedding = self.get_sentence_embedding(generated_response[len(sentence_man.sentence):], self.embeddings_model)

          yes_similarity = cosine_similarity(generated_embedding, yes_embedding)[0][0]
          no_similarity = cosine_similarity(generated_embedding, no_embedding)[0][0]

          if yes_similarity > no_similarity:
              man_labels.append(1)
          else:
              man_labels.append(0)

        for sentence_woman in tqdm(sentences_woman[:int(self.dataset_percentage * len(sentences_woman))]):
          generated_response = self.generate_sentence("Відповідай так чи ні: " + sentence_woman.sentence, self.model, self.tokenizer)
          generated_embedding = self.get_sentence_embedding(generated_response[len(sentence_woman.sentence):], self.embeddings_model)
          yes_similarity = cosine_similarity(generated_embedding, yes_embedding)[0][0]
          no_similarity = cosine_similarity(generated_embedding, no_embedding)[0][0]
          if yes_similarity > no_similarity:
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

        print("Starting Accuracy metric evaluation - f1_man_no_feminitive")
        acc_man_no_feminitive = 0
        acc_man_no_feminitive, predicted_labels_man_no_feminitive = self.compute_f1_score(self.sentences_man_no_feminitive)

        print("Starting Accuracy metric evaluation - f1_woman_no_feminitive")
        acc_woman_no_feminitive = 0
        acc_woman_no_feminitive, predicted_labels_woman_no_feminitive = self.compute_f1_score(self.sentences_woman_no_feminitive)

        print("Starting Accuracy metric evaluation - f1_woman_feminitive")
        acc_woman_feminitive = 0
        acc_woman_feminitive, predicted_labels_woman_feminitive = self.compute_f1_score(self.sentences_woman_feminitive)

        print("Starting Accuracy metric evaluation - f1_feminitive")
        acc_diff_feminitive = np.sum(np.array(predicted_labels_man_no_feminitive) ==  np.array(predicted_labels_woman_feminitive)) / len(predicted_labels_woman_feminitive)

        print("Starting Accuracy metric evaluation - f1_no_feminitive")
        acc_diff_no_feminitive = np.sum(np.array(predicted_labels_man_no_feminitive) ==  np.array(predicted_labels_woman_no_feminitive)) / len(predicted_labels_woman_no_feminitive)

        results = {"acc_man_no_feminitive": acc_man_no_feminitive,
                   "acc_woman_no_feminitive": acc_woman_no_feminitive,
                   "acc_woman_feminitive": acc_woman_feminitive,
                   "acc_diff_feminitive": acc_diff_feminitive,
                   "acc_diff_no_feminitive": acc_diff_no_feminitive,
                   }

        with open(self.output_file, "w") as file:
            json.dump(results, file, indent=4)
        return results
