import re
import numpy as np

CONTEXT_SIZE = 4  # n-1


def tokenize(text):
    """
    Turn raw text into a flat list of lowercase tokens.

    Keep it simple: lowercase the text, then use a regex to pull out either
    (a) runs of letters/apostrophes as words, or (b) individual punctuation
    marks .,!?;:" as their own tokens (so punctuation isn't glued onto words,
    and isn't silently dropped either -- it's part of the sequence the model
    sees, same as any word).

    Return: list of string tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z']+|[.,!?;:\"]", text)
    return tokens


def build_vocab(tokens, min_count=1):
    """
    Build a vocabulary from a list of tokens.

    Reserve index 0 for a special "<unk>" token (used for any word that
    occurs fewer than min_count times, so build_examples can map rare/unseen
    words to a shared fallback rather than crashing or silently growing the
    vocab unbounded).

    Return:
      word2idx: dict mapping word (str) -> integer index
      vocab_words: list of strings such that vocab_words[i] is the word for
                   index i (i.e. the inverse of word2idx) -- vocab_words[0]
                   should be "<unk>"
    """
    from collections import Counter

    word_counts = Counter(tokens)
    vocab_words = ["<unk>"] + [word for word, count in word_counts.items() if count >= min_count]
    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    return word2idx, vocab_words


def build_examples(tokens, word2idx, context_size=CONTEXT_SIZE):
    """
    Slide a window of size (context_size + 1) across the token sequence,
    turning it into training examples.

    For each position i (starting once you have context_size tokens before
    it), one training example is:
      context = the context_size tokens immediately before position i
      target  = the token at position i

    Map every token to its integer id via word2idx (use word2idx.get(t, 0)
    so out-of-vocab tokens map to <unk> = index 0 instead of crashing).

    Return: a 2D int array of shape (num_examples, context_size + 1), where
    each row is [context_id_1, ..., context_id_{n-1}, target_id].
    """

    examples = []
    for i in range(context_size, len(tokens)):
        context = tokens[i - context_size:i]
        target = tokens[i]
        context_ids = [word2idx.get(t, 0) for t in context]
        target_id = word2idx.get(target, 0)
        examples.append(context_ids + [target_id])
    return np.array(examples, dtype=np.int32)


def load_data(path="corpus_raw.txt", context_size=CONTEXT_SIZE, min_count=1):
    text = open(path).read()
    tokens = tokenize(text)
    word2idx, vocab_words = build_vocab(tokens, min_count=min_count)
    examples = build_examples(tokens, word2idx, context_size)
    return examples, word2idx, vocab_words, tokens


if __name__ == "__main__":
    examples, word2idx, vocab_words, tokens = load_data()
    print(f"Total tokens: {len(tokens)}")
    print(f"Vocabulary size: {len(vocab_words)}")
    print(f"Training examples: {examples.shape}")
    for row in examples[:3]:
        ctx = [vocab_words[i] for i in row[:-1]]
        tgt = vocab_words[row[-1]]
        print(f"  {ctx} -> {tgt}")