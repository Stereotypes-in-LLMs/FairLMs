import numpy as np
import pytest
import torch

from debias.soft_debias import (debias_word, gender_basis, legacy_debias_word, make_projectors,
                                project_out)
from debias.wrapper import DebiasWrapper

WORDS = dict(male_words="data/gender_words/male-words.csv", female_words="data/gender_words/female-words.csv")


def random_basis(d=32, k=5, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, k)))
    return q


def test_debias_word_removes_every_component():
    basis = random_basis()
    projectors = make_projectors(basis)
    w = np.random.default_rng(1).normal(size=32)
    out = debias_word(w, projectors)
    assert np.allclose(basis.T @ out, 0, atol=1e-10)
    # nothing outside the subspace changes
    assert np.allclose(out, w - basis @ (basis.T @ w))


def test_legacy_debias_word_reflects_first_component():
    basis = random_basis()
    w = np.random.default_rng(1).normal(size=32)
    out = legacy_debias_word(w, make_projectors(basis))
    assert np.isclose(basis[:, 0] @ out, -(basis[:, 0] @ w))  # flipped, not removed
    assert np.allclose(basis[:, 1:].T @ out, 0, atol=1e-10)


def test_project_out_matches_debias_word():
    basis = random_basis()
    w = np.random.default_rng(2).normal(size=(3, 32))
    expected = np.stack([debias_word(x, make_projectors(basis)) for x in w])
    got = project_out(torch.tensor(w), torch.tensor(basis)).numpy()
    assert np.allclose(got, expected, atol=1e-5)


def test_gender_basis_is_orthonormal_and_projection_vanishes(tiny):
    model, tok = tiny
    basis = gender_basis(model, tok, WORDS["male_words"], WORDS["female_words"], 0, 5)
    assert basis.shape == (model.config.hidden_size, 5)
    assert np.allclose(basis.T @ basis, np.eye(5), atol=1e-6)
    e = model.get_input_embeddings().weight[:10].detach()
    out = project_out(e, torch.tensor(basis, dtype=torch.float32))
    assert torch.allclose(out @ torch.tensor(basis, dtype=torch.float32), torch.zeros(10, 5), atol=1e-5)


@pytest.mark.parametrize("method,n_words", [("soft", 1), ("hard", 3)])
def test_target_tokens_found_inside_chat_template(tiny, method, n_words):
    model, tok = tiny
    w = DebiasWrapper(model=model, tokenizer=tok, method=method, **WORDS)
    sentence = "Кандидатка має досвід хірургічних операцій. Чи може вона отримати посаду хірургині?"
    text = tok.apply_chat_template([{"role": "user", "content": f"Дай відповідь.\n\n{sentence}"}],
                                   tokenize=False, add_generation_prompt=True)
    ids, offsets, idx = w.encode(text, "хірургині", sentence)
    decoded = tok.decode([ids[i] for i in idx]).lower()
    assert "хірургині" in decoded
    if method == "hard":
        assert "кандидатка" in decoded and "вона" in decoded
    # the debiased tokens are exactly those overlapping the target words: no template tokens
    assert all(offsets[i][0] >= text.rindex(sentence) for i in idx)
    # embeddings of those tokens have no component in the gender subspace, others are untouched
    ids_t = torch.tensor([ids])
    plain = model.get_input_embeddings()(ids_t)[0]
    deb = w.embed(ids_t, [idx])[0]
    assert torch.allclose((deb[idx] @ w.basis), torch.zeros(len(idx), 5), atol=1e-4)
    others = [i for i in range(len(ids)) if i not in idx]
    assert torch.equal(deb[others], plain[others])


def test_hard_rejects_sentence_without_pronoun(tiny):
    model, tok = tiny
    w = DebiasWrapper(model=model, tokenizer=tok, method="hard", **WORDS)
    with pytest.raises(AssertionError):
        w.encode("Кандидат має досвід. Посада хірурга.", "хірурга", "Кандидат має досвід. Посада хірурга.")
