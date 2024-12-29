from pydantic import BaseModel
from typing import Any
import torch
import numpy as np
import csv

def get_word_embedding(word, model, tokenizer, average = False):
    """
    Returns given word embedding
    """
    inputs = tokenizer(word,return_tensors="pt")
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1]

    if average:
        word_embedding = np.mean(np.array(hidden_states[0]), axis=0)
    else:
        word_embedding = hidden_states[0]

    return word_embedding

def cosine_similarity(u, v):
    '''
    Calculates the cosine similarity bettwen two vectors.
    It is the inner product of two vectors divided by product of their norms.
    '''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def calculate_gender_bias(neutral_word, male_gender, female_gender):
    '''
    Calculates the bias. If the sign is positive - the word is biased in male direction; if negative - in female direction.
    '''
    return cosine_similarity(neutral_word, male_gender) - cosine_similarity(neutral_word, female_gender)

def calculate_absolute_difference(neutral_word, male_gender, female_gender):
    '''
    Calculates the absolute difference between cosine similarities of two vectors with other.
    '''
    return abs(calculate_gender_bias(neutral_word, male_gender, female_gender))

def debias_word(word, projectors):
    '''
    Debiasing the word by taking projection.
    '''
    projection = projectors[0] @ word
    for i in range(0, len(projectors)):
        projection += projectors[i] @ word
    return word - projection

class SoftDebiasModelWrapper(BaseModel):

    model:Any
    tokenizer:Any
    gender_defining_man:str
    gender_defining_woman:str
    pca_component_s:int
    pca_component_e:int
    projectors:Any = []
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
        print(gender_word_pairs)
        for pair in gender_word_pairs:
            difference_vectors.append(get_word_embedding(pair[0], self.model, self.tokenizer,average = True) - get_word_embedding(pair[1], self.model, self.tokenizer, average = True))
        # Dimensionality reduction
        print(difference_vectors)
        X = np.array(difference_vectors)
        n =  4096
        X = X.T
        X -= np.mean(X, axis=0)
        C = np.cov(X, rowvar=False)
        l, principal_axes = np.linalg.eig(C)
        idx = l.argsort()[::-1]
        l, principal_axes = l[idx], principal_axes[:, idx]
        principal_components = X.dot(principal_axes)
        print(principal_components.shape)
        PCA_k = principal_components[:, self.pca_component_s : self.pca_component_e]
        # finding projectors
        self.projectors = []
        for i in range(PCA_k.shape[-1]):
            a = PCA_k[:, i]
            self.projectors.append(np.outer(a, a) / np.inner(a, a))
    
    def debias_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        token_embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        # tokens = inputs['input_ids'][0].tolist()
        # question_mark = tokens.index(29584)
        # for i in range(question_mark, 0, -1):
        #     current_token = self.tokenizer.convert_ids_to_tokens(tokens[i])
        #     if "▁" in current_token:
        #         break
        #     start = i
        #     print(self.tokenizer.convert_ids_to_tokens(tokens[start:question_mark]))
        # for i in range(start, question_mark, 1):
        #     token_embeddings[0][i] = torch.tensor(debias_word((token_embeddings[0][i]).tolist(), self.projectors))
        return inputs['attention_mask'], token_embeddings

    def forward(self, tokens):
        modified_embeddings = self.debias_embeddings(tokens)
        outputs = self.model(attention_mask=modified_embeddings[0],
                    inputs_embeds=modified_embeddings[1])
        return outputs
    
    def __call__(self, tokens):
        return self.forward(tokens)

