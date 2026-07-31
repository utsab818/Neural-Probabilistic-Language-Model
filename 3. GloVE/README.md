# GloVe (from scratch, NumPy)

A from-scratch implementation of Pennington, Socher & Manning's GloVe (2014)
— a follow-up to NPLM and word2vec implementations. Pure NumPy: co-occurrence
counting, forward pass, and hand-derived backprop, no autograd.

## Idea

Instead of learning embeddings by repeatedly sampling local windows (like
word2vec), count word co-occurrences across the **whole corpus once**, then
fit embeddings directly to those global counts via weighted least squares.

## Model

```
diff = w_i . w̃_j + b_i + b̃_j - log(X_ij)
f(X_ij) = (X_ij/x_max)^0.75   if X_ij < x_max,  else 1
Loss = sum over observed pairs of:  f(X_ij) * diff^2
```

`X_ij` = co-occurrence count between word `i` and word `j`, weighted by
`1/distance` within a window. Two vectors per word (`w`, `w̃`) — needed so
a word's co-occurrence with itself doesn't tie its embedding to its own
squared norm.

## Files

- `model.py` — the `GloVe` class (`weight_fn`, `forward`, `loss`, `backward`, `step`)
- `data.py` — tokenizing, vocab, and the co-occurrence matrix builder
- `grad_check.py` — verifies backprop against numerical finite differences
- `train.py` — training loop, saves `trained_glove.npz` + `vocab_glove.pkl`
- `corpus_raw.txt` — training text (*Alice's Adventures in Wonderland*, public domain)

## Usage

```bash
python3 grad_check.py   # verify backprop
python3 train.py        # build co-occurrence matrix, train, save
```

## Result

1,118-word vocab, 28,431 nonzero co-occurrence pairs: loss 0.051 → 0.034
over 50 epochs. Embeddings are noisier than the NPLM/word2vec results on
this corpus as expected, since GloVe's signal comes from co-occurrence
*statistics*, which need much more text than ~7,000 words to be reliable.

## Reference

Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors
for Word Representation. *EMNLP*.