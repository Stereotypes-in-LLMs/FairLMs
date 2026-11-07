"""
Embedding-projection debiasing.

A gender subspace is estimated from difference vectors of gender-defining word pairs (PCA, components
[pca_s, pca_e)), and the input embeddings of selected tokens are projected onto its orthogonal
complement. The "soft" method debiases the profession tokens; the "hard" method additionally debiases
"кандидат"/"кандидатка" and "він"/"вона". Note: neither is the hard debias of Bolukbasi et al. (2016)
(neutralize + equalize); both are the same projection applied to different token sets.
"""
import csv

import numpy as np
import torch


def get_word_embedding(word, model, tokenizer):
    """
    Returns the mean input embedding of the word's tokens (without BOS or other special tokens).
    """
    ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")["input_ids"]
    token_embeddings = model.get_input_embeddings()(ids.to(model.device))
    return token_embeddings[0].float().mean(dim=0).detach().cpu().numpy()


def read_words(path):
    words = []
    with open(path, newline="") as csvfile:
        for row in csv.reader(csvfile):
            words.extend(word.strip() for word in row if word.strip())
    return words


def gender_basis(model, tokenizer, male_words_path, female_words_path, pca_s, pca_e):
    """
    Orthonormal basis (d x k) of the gender subspace: principal directions [pca_s, pca_e) of the
    female - male difference vectors.
    """
    male, female = read_words(male_words_path), read_words(female_words_path)
    assert len(male) == len(female), "gender word lists must be paired line by line"
    differences = [get_word_embedding(f, model, tokenizer) - get_word_embedding(m, model, tokenizer)
                   for m, f in zip(male, female)]

    # Same PCA as in v1 (directions in embedding space = X @ eigenvectors of the pair covariance),
    # with eigh instead of eig: C is symmetric, so eigenpairs are real and sorted reliably.
    X = np.array(differences, dtype=np.float64).T
    X -= np.mean(X, axis=0)
    C = np.cov(X, rowvar=False)
    eigenvalues, principal_axes = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    principal_components = X @ principal_axes[:, idx]
    basis = principal_components[:, pca_s:pca_e]
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def make_projectors(basis):
    return [np.outer(a, a) for a in basis.T]


def debias_word(word, projectors):
    """
    Debiasing the word by removing its projection on every gender component.

    v1 initialised the sum with projectors[0] @ word and then added all projectors again,
    i.e. returned w - 2 P_0 w - P_1 w - ..., which reflects the first component instead of removing it.
    """
    projection = sum(P @ word for P in projectors)
    return word - projection


def project_out(embeddings, basis):
    """
    Batched equivalent of debias_word for mutually orthogonal components: e - U U^T e.
    embeddings: (..., d) tensor, basis: (d, k) tensor.
    """
    dtype = embeddings.dtype
    e = embeddings.to(basis.dtype)
    return (e - (e @ basis) @ basis.T).to(dtype)


def legacy_debias_word(word, projectors):
    """The v1 implementation, kept only so tests can document the bug."""
    projection = projectors[0] @ word
    for i in range(0, len(projectors)):
        projection += projectors[i] @ word
    return word - projection


def basis_tensor(basis, device):
    return torch.tensor(basis, dtype=torch.float32, device=device)
