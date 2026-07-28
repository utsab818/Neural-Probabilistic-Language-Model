# Neural Probabilistic Language Model (from scratch, NumPy)

A from-scratch implementation of Bengio et al.'s 2003 paper
[*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
— the ancestor of word embeddings and modern LLMs. Pure NumPy: forward pass,
backprop, and SGD all hand-derived, no autograd.

## Idea

Instead of treating words as unrelated symbols (like classic n-gram models
do), each word gets a learned dense vector (embedding). Words used in
similar contexts end up with similar vectors, purely as a side effect of
learning to predict the next word.

## Architecture

```
x = concat(C[w_1], ..., C[w_{n-1}])   # embed context words, concatenate
a = H @ x + d
z = tanh(a)                            # nonlinearity
o = U @ z + b
p = softmax(o)                         # probability over vocabulary
L = -log(p[target])
```

## Files

- `model.py` — the `NPLM` class (init, forward, loss, backward, SGD step)
- `data.py` — tokenizes text, builds vocab, makes (context, target) examples
- `train.py` — training loop, saves `trained_params.npz` + `vocab.pkl`
- `explore.py` — nearest-neighbor embeddings + text generation
- `corpus_raw.txt` — training text (*Alice's Adventures in Wonderland*, public domain)

## Usage

```bash
python3 train.py        # train the model
python3 explore.py      # inspect embeddings, generate text
```

## Result

Trained on ~7,000 tokens (1,119-word vocab): validation perplexity drops
from 847 to 160 over 60 epochs. Nearest neighbors of "said" come out as
`inquired`, `muttered`, `shouted`, `cried` — speech verbs clustering together with zero explicit supervision.


![alt text](output/image.png)

## Reference

Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural
Probabilistic Language Model. *JMLR*, 3, 1137–1155.