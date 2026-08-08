import numpy as np

# Toy task: reverse a short sequence of random tokens. Standard sanity check
# for seq2seq models, since it doesn't need a real parallel translation
# corpus, but genuinely requires the model to use cross-attention correctly
# (the decoder needs the FULL source sequence to produce the reversed output).

PAD, BOS, EOS = 0, 1, 2
VOCAB_OFFSET = 3  # token ids 3..vocab_size-1 are "content" tokens


def make_example(rng, seq_len, vocab_size):
    content = rng.integers(VOCAB_OFFSET, vocab_size, size=seq_len)
    src = content
    reversed_content = content[::-1]
    tgt_full = np.concatenate([[BOS], reversed_content, [EOS]])
    tgt_input = tgt_full[:-1]   # BOS, r1, r2, ..., rn
    tgt_output = tgt_full[1:]   # r1, r2, ..., rn, EOS
    return src, tgt_input, tgt_output


def make_dataset(n_examples, seq_len, vocab_size, seed=0):
    rng = np.random.default_rng(seed)
    return [make_example(rng, seq_len, vocab_size) for _ in range(n_examples)]
