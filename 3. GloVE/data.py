import re
import numpy as np
from collections import Counter, defaultdict

WINDOW_SIZE = 4


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z']+|[.,!?;:\"]", text)


def build_vocab(tokens, min_count=1):
    counts = Counter(tokens)
    vocab_words = [w for w, c in counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    return word2idx, vocab_words


def build_cooccurrence(tokens, word2idx, window_size=WINDOW_SIZE):
    """
    Build the co-occurrence matrix X as a dict: {(i, j): weighted_count}.

    For every position t in the token sequence, look at every OTHER token
    within window_size positions on either side. For each such neighbor at
    distance d (d = 1, 2, ..., window_size), add 1/d to X[(i,j)], where i is
    the vocabulary id of tokens[t] and j is the vocabulary id of the
    neighbor.

    Return: a dict mapping (i, j) -> float (the accumulated weighted count).
    Use a defaultdict(float) so you can += into entries that don't exist yet
    without a KeyError.

    Note: since co-occurrence is symmetric in principle (X_ij should equal
    X_ji over the whole corpus), you do NOT need to add both (i,j) and (j,i)
    separately for each single occurrence -- if your loop naturally visits
    every token as the "center" at some point (which it will, since you loop
    over every position t), both directions get accumulated automatically
    across the full pass. Just add (i,j) for center=i, neighbor=j on each
    step -- don't double-add symmetric entries manually, or you'll double
    count.
    """
    cooc = defaultdict(float)
    for t, center in enumerate(tokens):
        i = word2idx[center]
        for d in range(1, window_size + 1):
            if t - d >= 0:
                neighbor = tokens[t - d]
                j = word2idx[neighbor]
                cooc[(i, j)] += 1.0 / d
            if t + d < len(tokens):
                neighbor = tokens[t + d]
                j = word2idx[neighbor]
                cooc[(i, j)] += 1.0 / d
    return cooc


def load_data(path="corpus_raw.txt", window_size=WINDOW_SIZE, min_count=1):
    text = open(path).read()
    tokens = tokenize(text)
    word2idx, vocab_words = build_vocab(tokens, min_count=min_count)
    cooc = build_cooccurrence(tokens, word2idx, window_size)
    return cooc, word2idx, vocab_words


if __name__ == "__main__":
    cooc, word2idx, vocab_words = load_data()
    print(f"Vocab size: {len(vocab_words)}")
    print(f"Nonzero co-occurrence pairs: {len(cooc)}")
    # sanity check: symmetry
    sample_pairs = list(cooc.items())[:3]
    for (i, j), val in sample_pairs:
        reverse = cooc.get((j, i), 0.0)
        print(f"  X[{vocab_words[i]!r},{vocab_words[j]!r}] = {val:.3f}  |  "
              f"X[{vocab_words[j]!r},{vocab_words[i]!r}] = {reverse:.3f}")