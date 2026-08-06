import re
import numpy as np
from collections import Counter

PAD_CHAR = 0  # reserved char id for padding short words


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z']+|[.,!?;:\"]", text)


def build_char_vocab(tokens):
    chars = sorted(set("".join(tokens)))
    char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for PAD
    return char2idx


def build_word_vocab(tokens, min_count=1):
    counts = Counter(tokens)
    vocab_words = ["<unk>"] + [w for w, c in counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    return word2idx, vocab_words


def word_to_char_ids(word, char2idx, min_len):
    ids = [char2idx.get(c, 0) for c in word]
    while len(ids) < min_len:
        ids.append(PAD_CHAR)  # right-pad short words so CharCNN always has >=1 window
    return np.array(ids, dtype=np.int32)


def build_sentences(tokens, sentence_len=20):
    """Chop the token stream into fixed-length chunks (simple stand-in for
    real sentence boundaries, since our corpus tokenizer doesn't track them
    beyond '.', '!', '?' as plain punctuation tokens)."""
    sentences = []
    for i in range(0, len(tokens) - sentence_len, sentence_len):
        sentences.append(tokens[i:i + sentence_len])
    return sentences


def load_data(path="corpus_raw.txt", min_char_len=3, sentence_len=20, min_count=1):
    text = open(path).read()
    tokens = tokenize(text)
    char2idx = build_char_vocab(tokens)
    word2idx, vocab_words = build_word_vocab(tokens, min_count=min_count)
    sentences = build_sentences(tokens, sentence_len)

    sentences_char_ids = []
    sentences_word_ids = []
    for sent in sentences:
        char_id_seqs = [word_to_char_ids(w, char2idx, min_char_len) for w in sent]
        word_ids = np.array([word2idx.get(w, 0) for w in sent], dtype=np.int32)
        sentences_char_ids.append(char_id_seqs)
        sentences_word_ids.append(word_ids)

    return sentences_char_ids, sentences_word_ids, char2idx, word2idx, vocab_words


if __name__ == "__main__":
    sc, sw, char2idx, word2idx, vocab_words = load_data()
    print(f"Char vocab size: {len(char2idx) + 1}")
    print(f"Word vocab size: {len(vocab_words)}")
    print(f"Number of sentences: {len(sc)}")
    print(f"First sentence word ids: {sw[0]}")
    print(f"First word's char ids: {sc[0][0]}")
