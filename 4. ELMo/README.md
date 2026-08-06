# ELMo (Deep Contextualized Word Representations) — from scratch, NumPy

A from-scratch implementation of Peters et al.'s ELMo (2018), following
NPLM, word2vec, and GloVe implementations. Pure NumPy: LSTM with full
backprop-through-time, a character-level CNN, highway networks, and a
bidirectional language model, all hand-derived, no autograd.

## Idea

Every previous model gives each word ONE fixed vector, reused everywhere —
"bank" gets the same representation whether it means "riverbank" or
"financial bank." ELMo fixes this by building a genuinely **contextual**
representation: a deep bidirectional LSTM language model, where a word's
final representation is a learned mixture of *every layer's* hidden state
at that position — layers that, by construction, depend on the whole
surrounding sentence.

## Architecture

```
word -> characters -> CharCNN (filters + max-pooling) -> Highway network -> x_t
x_t sequence -> forward LSTM  -> H_fwd_t   (predicts word_{t+1})
x_t sequence -> backward LSTM -> H_bwd_t   (predicts word_{t-1}, tied CharCNN/output)

ELMo_word = gamma * sum_j( softmax(s)_j * layer_j )   (learned per downstream task)
```

- **LSTM**: `f,i,o` gates (sigmoid) + candidate (tanh); cell state updated by
  *gated addition* (`c_t = f_t⊙c_{t-1} + i_t⊙c̃_t`) rather than forced
  squash-and-overwrite — this is what avoids the vanishing gradient problem
  vanilla RNNs have.
- **CharCNN**: slides learned filters across a word's characters, max-pools
  over positions -> fixed-size, vocabulary-free word representation from
  spelling alone (handles rare/unseen words).
- **Highway network**: `y = (1-g)*x + g*tanh(Wx+b)`, a learned gated
  shortcut — same vanishing-gradient fix as the LSTM, applied across depth
  instead of time.
- **ScalarMix**: the actual "deep contextualized" combination — learned
  per-layer weights + a rescaling `gamma`, so different downstream tasks can
  lean on different layers.

## Files

- `lstm.py` — `LSTMCell`: single-step + full-sequence forward/backward (BPTT)
- `charcnn.py` — `CharCNN`: filters, max-pooling, forward/backward
- `highway.py` — `Highway`: gated shortcut layer, forward/backward
- `model.py` — `BiLM`: wires CharCNN+Highway+both LSTMs+tied output softmax
  into the full bidirectional language model
- `scalar_mix.py` — `ScalarMix`: the layer-combination mechanism
- `data.py` — character vocab, word vocab, sentence chunking
- `grad_check_*.py` — numerical gradient checks for every component
  (LSTM single-step, full BPTT, CharCNN, Highway, ScalarMix, and the full
  end-to-end model)
- `train.py` — training loop, saves all trained parameters
- `demo_elmo.py` — loads the trained model, shows "bank" getting a
  different `h_1` (LSTM layer) but identical `h_0` (CharCNN layer) across
  two different sentences, then combines them with `ScalarMix`
- `corpus_raw.txt` — training text (*Alice's Adventures in Wonderland*, public domain)

## Usage

```bash
python3 grad_check_lstm.py         # single-timestep LSTM backprop
python3 grad_check_bptt.py         # full backprop-through-time
python3 grad_check_charcnn.py      # CharCNN backprop
python3 grad_check_highway.py      # highway network backprop
python3 grad_check_scalar_mix.py   # layer-combination backprop
python3 grad_check_model.py        # the WHOLE model, end to end
python3 train.py                   # train the biLM
python3 demo_elmo.py               # see contextual vs. static layers in action
```

## Result

35-character vocab, 1,119-word vocab, 350 (chunked) sentences: loss 12.97 →
11.03 over 5 epochs. Perplexity is high and "bank" barely differentiates
between contexts (cosine similarity ~0.9999999) — this model is trained on
~7,000 words for a few minutes; the real ELMo paper trains on a billion-word
corpus for days. The architecture and every gradient are verified correct
(see the `grad_check_*` scripts); getting *strong* contextual
differentiation needs real scale.

## Reference

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., &
Zettlemoyer, L. (2018). Deep contextualized word representations. *NAACL*.
