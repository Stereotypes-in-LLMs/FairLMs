"""
One wrapper for all inference-time methods.

* none   - plain model;
* prompt - debias instruction prepended to the text (handled when the prompt is built);
* soft   - projection debiasing of the profession tokens;
* hard   - projection debiasing of the profession tokens + "кандидат(ка)" + "він"/"вона".

The words to debias are passed explicitly (`job_form` column of the dataset) and located in the
tokenized text through offset mappings, so the chat template cannot break the lookup.
"""
import re

import torch

from debias.soft_debias import basis_tensor, gender_basis, project_out

METHODS = ("none", "prompt", "soft", "hard")
HARD_WORDS = ("кандидат", "кандидатка", "він", "вона")


def find_word_spans(text, words, start=0, end=None):
    """Char spans of whole-word, case-insensitive occurrences of each word in text[start:end]."""
    end = len(text) if end is None else end
    spans = {}
    for word in words:
        pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
        spans[word] = [(m.start() + start, m.end() + start) for m in pattern.finditer(text[start:end])]
    return spans


def char_spans_to_tokens(offsets, char_spans):
    """Indices of tokens whose character range overlaps any of the spans."""
    idx = []
    for i, (s, e) in enumerate(offsets):
        if e > s and any(s < ce and e > cs for cs, ce in char_spans):
            idx.append(i)
    return idx


class DebiasWrapper:
    def __init__(self, model, tokenizer, method="none", debias_prompt=None, male_words=None,
                 female_words=None, pca_s=0, pca_e=5):
        assert method in METHODS, f"unknown method {method}"
        self.model, self.tokenizer, self.method = model, tokenizer, method
        self.debias_prompt = debias_prompt
        self.basis = None
        if method in ("soft", "hard"):
            basis = gender_basis(model, tokenizer, male_words, female_words, pca_s, pca_e)
            self.basis = basis_tensor(basis, model.device)

    @property
    def device(self):
        return self.model.device

    def with_prompt(self, text, sep=" "):
        return f"{self.debias_prompt}{sep}{text}" if self.method == "prompt" else text

    def debias_char_spans(self, text, job_form, start, end):
        """Char spans to debias inside text[start:end] (the dataset sentence)."""
        if self.method not in ("soft", "hard"):
            return []
        found = find_word_spans(text, [job_form], start, end)
        assert len(found[job_form]) == 1, f"profession form {job_form!r} found {len(found[job_form])} times"
        spans = list(found[job_form])
        if self.method == "hard":
            extra = find_word_spans(text, [w for w in HARD_WORDS if w != job_form.lower()], start, end)
            hits = [s for v in extra.values() for s in v]
            assert len(hits) == 2, f"expected candidate noun + pronoun, got {extra}"
            spans += hits
        return spans

    def encode(self, text, job_form=None, sentence=None):
        """
        Tokenizes text (special tokens must already be in it, e.g. from the chat template) and returns
        input ids plus the token indices to debias (those of `sentence`'s target words).
        """
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        debias_idx = []
        if self.method in ("soft", "hard"):
            start = text.rindex(sentence)
            spans = self.debias_char_spans(text, job_form, start, start + len(sentence))
            debias_idx = char_spans_to_tokens(enc["offset_mapping"], spans)
        return enc["input_ids"], enc["offset_mapping"], debias_idx

    def embed(self, input_ids, debias_idx):
        """
        input_ids: (B, T) tensor; debias_idx: per row, token positions (in the padded batch) to debias.
        """
        embeddings = self.model.get_input_embeddings()(input_ids.to(self.device))
        if self.basis is not None:
            for row, positions in enumerate(debias_idx):
                if positions:
                    pos = torch.tensor(positions, device=self.device)
                    embeddings[row, pos] = project_out(embeddings[row, pos], self.basis)
        return embeddings

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, debias_idx, position_ids=None, **kwargs):
        embeddings = self.embed(input_ids, debias_idx)
        return self.model(inputs_embeds=embeddings, attention_mask=attention_mask.to(self.device),
                          position_ids=None if position_ids is None else position_ids.to(self.device),
                          **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, debias_idx, **kwargs):
        embeddings = self.embed(input_ids, debias_idx)
        return self.model.generate(inputs_embeds=embeddings, attention_mask=attention_mask.to(self.device),
                                   **kwargs)
