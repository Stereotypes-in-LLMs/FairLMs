from enum import Enum

class TokensDebiasType(Enum):
    LAST_WORD = "last_word"
    FIRST_SENTENCE_WORD = "first_sentence_word"
    NONE = "none"
