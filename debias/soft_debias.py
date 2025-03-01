from pydantic import BaseModel
from typing import Any
import torch
import numpy as np
import csv
import re
from sklearn.metrics.pairwise import cosine_similarity
from utils import TokensDebiasType


def get_word_embedding(word, model, tokenizer, average = False):
    """
    Returns given word embedding
    """
    inputs = tokenizer(word, return_tensors="pt")
    token_embeddings = model.get_input_embeddings()(inputs['input_ids'].to("cuda"))
    word_embedding = np.mean(np.array(token_embeddings[0].cpu().detach().numpy()), axis=0)
    return word_embedding

def calculate_gender_bias(neutral_word, male_gender, female_gender):
    """
    Calculates the bias. If the sign is positive - the word is biased in male direction; if negative - in female direction.
    """
    return cosine_similarity(neutral_word, male_gender) - cosine_similarity(neutral_word, female_gender)

def calculate_absolute_difference(neutral_word, male_gender, female_gender):
    """
    Calculates the absolute difference between cosine similarities of two vectors with other.
    """
    return abs(calculate_gender_bias(neutral_word, male_gender, female_gender))

def debias_word(word, projectors):
    """
    Debiasing the word by taking projection.
    """

    projection = projectors[0] @ word
    for i in range(0, len(projectors)):
        projection += projectors[i] @ word
    return word - projection

def find_subset_indices(lst, subset):
    n = len(subset)
    for i in range(len(lst) - n + 1):
        if lst[i:i+n] == subset:
            return i + len(subset)
    return -1


class SoftDebiasModelWrapper(BaseModel):
    model:Any
    model_name: str
    tokenizer:Any
    gender_defining_man:str
    gender_defining_woman:str
    pca_component_s:int
    pca_component_e:int
    projectors:Any = []
    hard_like:bool = False
    UNCONDITIONAL_START_TOKEN: str = "<s>"
    
    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.make_projectors()
    
    def make_projectors(self):
        gender_words_female = []
        gender_words_male = []

        with open(self.gender_defining_woman, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                for word in row:
                    gender_words_female.append(word.strip())

        with open(self.gender_defining_man, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                for word in row:
                    gender_words_male.append(word.strip())

        gender_word_pairs = [[female, male] for female, male in zip(gender_words_female, gender_words_male)]

        difference_vectors = []
        for pair in gender_word_pairs:
            difference_vectors.append(get_word_embedding(pair[0], self.model, self.tokenizer, average = True) -
                                       get_word_embedding(pair[1], self.model, self.tokenizer, average = True))
        
        # Dimensionality reduction
        X = np.array(difference_vectors)
        X = X.T
        X -= np.mean(X, axis=0)
        C = np.cov(X, rowvar=False)
        l, principal_axes = np.linalg.eig(C)
        idx = l.argsort()[::-1]
        l, principal_axes = l[idx], principal_axes[:, idx]
        principal_components = X.dot(principal_axes)
        PCA_k = principal_components[:, self.pca_component_s : self.pca_component_e]
        
        # finding projectors
        self.projectors = []
        for i in range(PCA_k.shape[-1]):
            a = PCA_k[:, i]
            self.projectors.append(np.outer(a, a) / np.inner(a, a))

    def is_word_in_sentence(self, word, sentence):
        word = word.lower()
        sentence = sentence = re.sub(r'[.,?!]', '', sentence.lower())
        words = sentence.split()
        return word in words

    def debias_embeddings(self, text, token_debias_type: TokensDebiasType = TokensDebiasType.LAST_WORD):

        debias_words = []
        if self.hard_like:
            debias_words = ["кандидат", "кандидатка", "він", "вона"]
        if token_debias_type == TokensDebiasType.FIRST_SENTENCE_WORD:
            word = text.split(". Посаду ")[-1]
            word = word.split(" ")[0]
            debias_words.append(word)
            
        elif token_debias_type == TokensDebiasType.LAST_WORD:
            word = text.strip().split(" ")
            index = word.index("посаду")
            word = word[index + 1][:-1]
            debias_words.append(word)
            
        inputs = self.tokenizer(text, return_tensors="pt")
        token_embeddings = self.model.get_input_embeddings()(inputs['input_ids'].to("cuda"))

        tokens = inputs['input_ids'][0].tolist()
        debias_indexes = []
        found_words = 0
        if text != self.UNCONDITIONAL_START_TOKEN:
            for word in debias_words:
                left_part = 0
                right_part = 0
                not_found = False
                for left in range(len(tokens)):
                    if self.is_word_in_sentence(word, self.tokenizer.decode(tokens[left:])):
                        continue
                    else:
                        if left == 0:
                            not_found = True
                            break
                        left_part = left - 1
                        break
                if not_found:
                    print("Not Found", word)
                    continue
                for right in range(len(tokens), left - 1, -1):
                    if self.is_word_in_sentence(word,self.tokenizer.decode(tokens[left_part:right])):
                        continue
                    else:
                        if right == len(tokens):
                            right_part = right 
                        else:
                            right_part = right + 1
                        break
                assert word.lower() in self.tokenizer.decode(tokens[left_part:right_part]).lower()
                assert word.lower() not in self.tokenizer.decode(tokens[left_part+1:right_part]).lower()
                assert word.lower() not in self.tokenizer.decode(tokens[left_part:right_part-1]).lower()
                found_words += 1
                print("Found", self.tokenizer.decode(tokens[left_part:right_part]).lower())
                debias_indexes.append((left_part, right_part))
            if self.hard_like:
                assert found_words == 3
            else:
                assert found_words == 1
        for pair in debias_indexes:
            for i in range(pair[1] - pair[0]):
                token_embeddings[0][pair[0] + i] = torch.tensor(debias_word((token_embeddings[0][pair[0] + i]).tolist(), self.projectors))
        return inputs['attention_mask'], token_embeddings

    def forward(self, tokens, token_debias_type, output_hidden_states: bool = False):
        modified_embeddings = self.debias_embeddings(tokens, token_debias_type = token_debias_type)
        outputs = self.model(attention_mask=modified_embeddings[0],
                    inputs_embeds=modified_embeddings[1], output_hidden_states = output_hidden_states)
        return outputs
    
    def __call__(self, tokens, output_hidden_states: bool = False, token_debias_type: TokensDebiasType = TokensDebiasType.NONE):
        return self.forward(tokens, token_debias_type, output_hidden_states = output_hidden_states)