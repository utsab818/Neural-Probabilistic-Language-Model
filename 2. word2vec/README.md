# word2vec (Skip-gram + CBOW, from scratch, NumPy)

A from-scratch implementation of Mikolov et al.'s word2vec (2013), built as
a follow-up to an [NPLM implementation](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
Pure NumPy: forward pass, backprop, and negative sampling all hand-derived,
no autograd.

## Idea

Drop the NPLM's hidden layer (only embedding quality matters here, not full
grammatical modeling) and replace the expensive full-vocabulary softmax with
**negative sampling**: instead of scoring every word in the vocabulary,
just learn to tell real `(context, word)` pairs apart from a handful of
random fake ones. Two architectures:

- **Skip-gram**: one center word -> predict each surrounding context word independently
- **CBOW**: average several context words -> predict the one target word

## Architecture

```
Skip-gram: x = C_in[center_word]
CBOW:      x = mean(C_in[context_words])

score(w) = x . C_out[w]
p_real   = sigmoid(score(target))
p_fake_i = sigmoid(score(negative_i))     for k negative samples

L = -log(p_real) - sum_i( log(1 - p_fake_i) )
```

Two separate embedding tables, `C_in` and `C_out` -- needed so a word
sampled as its own negative example doesn't force a self-dot-product
contradiction.

## Files

- `model.py` — the `Word2Vec` class (`mode="skipgram"` or `mode="cbow"`)
- `data.py` — tokenizing, vocab, Skip-gram/CBOW pair generation, `count^0.75` negative sampling distribution
- `grad_check.py` / `grad_check_cbow.py` — verify backprop against numerical finite differences
- `train.py` / `train_cbow.py` — training loops, save `trained_w2v.npz` + `vocab_w2v.pkl`
- `explore.py` — nearest-neighbor embeddings
- `corpus_raw.txt` — training text (*Alice's Adventures in Wonderland*, public domain)

## Usage

```bash
python3 grad_check.py         # verify skip-gram backprop
python3 grad_check_cbow.py    # verify cbow backprop
python3 train.py              # train skip-gram
python3 train_cbow.py         # train cbow
python3 explore.py            # inspect skip-gram embeddings
```

## Result

Skip-gram (28,030 training pairs): loss 4.16 -> 2.73 over 20 epochs.
CBOW (7,005 training pairs): loss 4.16 -> 3.36 over 20 epochs — fewer
training pairs from the same corpus, since CBOW blends context words
together instead of predicting each one separately, matching the
paper's claim that Skip-gram gives more (and better, for rare words)
training signal per corpus.

## Reference

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient
Estimation of Word Representations in Vector Space. *arXiv:1301.3781*.
