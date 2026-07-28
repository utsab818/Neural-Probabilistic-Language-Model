import re
import numpy as np
from collections import Counter

WINDOW_SIZE = 2  # words on each side of the center word (so context size = 2*WINDOW_SIZE)


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z']+|[.,!?;:\"]", text)


def build_vocab(tokens, min_count=1):
    counts = Counter(tokens)
    vocab_words = ["<unk>"] + [w for w, c in counts.items() if c >= min_count]
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    # word_counts, aligned with vocab_words/word2idx indices (index 0 = <unk>,
    # give it count 0 so it never gets sampled as a negative example)
    word_counts = np.array([0] + [counts[w] for w in vocab_words[1:]], dtype=np.float64)
    return word2idx, vocab_words, word_counts


def build_skipgram_pairs(tokens, word2idx, window_size=WINDOW_SIZE):
    """
    Slide through the token sequence; for each position i, treat tokens[i] as
    the center word, and pair it with every token within window_size on
    either side as a separate (center, context) example.

    Return: two 1D int arrays, center_ids and context_ids, same length --
    row j is one (center, context) pair.

    Edge handling: near the start/end of the token sequence, just use
    whatever window actually exists (don't pad -- unlike the NPLM, there's
    no fixed-size input here, so a shorter window at the edges is fine).
    """
    # TODO: implement
    center_ids = []
    context_ids = []
    for i in range(len(tokens)):
        center_word = tokens[i]
        center_id = word2idx.get(center_word, 0)
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if j != i:
                context_word = tokens[j]
                context_id = word2idx.get(context_word, 0)
                center_ids.append(center_id)
                context_ids.append(context_id)
    return np.array(center_ids, dtype=np.int32), np.array(context_ids, dtype=np.int32)


def build_cbow_pairs(tokens, word2idx, window_size=WINDOW_SIZE):
    """
    Mirror image of build_skipgram_pairs: for each position i with a FULL
    window on both sides (so every example has the same fixed window size,
    since CBOW needs a fixed-shape (window,) array to average), collect the
    surrounding context words as input_ids and tokens[i] as the target.
    """
    input_ids = []
    target_ids = []
    for i in range(window_size, len(tokens) - window_size):
        context = tokens[i - window_size:i] + tokens[i + 1:i + window_size + 1]
        context_ids = [word2idx.get(t, 0) for t in context]
        target_id = word2idx.get(tokens[i], 0)
        input_ids.append(context_ids)
        target_ids.append(target_id)
    return np.array(input_ids, dtype=np.int32), np.array(target_ids, dtype=np.int32)


def build_negative_sampler(word_counts, power=0.75):
    """
    word_counts: 1D array, word_counts[i] = raw count of vocab word i
                 (index 0, <unk>, should have count 0 and never be sampled)

    Return: a 1D probability array `neg_probs`, same length as word_counts,
    where neg_probs[i] = count(i)^power / sum_j(count(j)^power).

    (This is what you'll pass as `p=` to rng.choice(..., p=neg_probs) later,
    to draw negative samples according to the paper's distribution instead
    of uniformly.)
    """
    # TODO: implement
    counts_powered = np.power(word_counts, power)
    neg_probs = counts_powered / np.sum(counts_powered)
    return neg_probs


def load_data(path="corpus_raw.txt", window_size=WINDOW_SIZE, min_count=1):
    text = open(path).read()
    tokens = tokenize(text)
    word2idx, vocab_words, word_counts = build_vocab(tokens, min_count=min_count)
    center_ids, context_ids = build_skipgram_pairs(tokens, word2idx, window_size)
    neg_probs = build_negative_sampler(word_counts)
    return center_ids, context_ids, neg_probs, word2idx, vocab_words


if __name__ == "__main__":
    center_ids, context_ids, neg_probs, word2idx, vocab_words = load_data()
    print(f"Vocab size: {len(vocab_words)}")
    print(f"Skip-gram pairs: {len(center_ids)}")
    for i in range(5):
        print(f"  {vocab_words[center_ids[i]]!r} -> {vocab_words[context_ids[i]]!r}")
    print(f"neg_probs sums to: {neg_probs.sum():.6f} (should be ~1.0)")
    print(f"neg_probs[<unk>] = {neg_probs[0]} (should be 0)")
    top5 = np.argsort(-neg_probs)[:5]
    print("Most likely negative samples:", [vocab_words[i] for i in top5])
