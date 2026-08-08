# Attention Is All You Need (Transformer, from scratch, NumPy)

A from-scratch implementation of Vaswani et al.'s Transformer (2017). Pure NumPy:
scaled dot-product attention, multi-head attention, masking, cross-attention,
layer norm, and the full encoder-decoder stack, all hand-derived, no autograd.


## Idea

RNNs/LSTMs process a sequence one step at a time — `h_t` can't be computed
until `h_{t-1}` exists, which wastes parallel hardware. The Transformer
replaces recurrence with **attention**: every word directly looks at every
other word in the sentence, all at once, and decides which ones matter.
No sequential dependency, fully parallel.

## Architecture

```
Query = x @ W_Q,  Key = x @ W_K,  Value = x @ W_V
score(t,s) = (Query_t . Key_s) / sqrt(d_k)        <- scaled, to stop softmax
                                                       saturating as d_k grows
attn_weights = softmax(score)
y_t = sum_s( attn_weights[t,s] * Value_s )

MultiHead = concat(head_1, ..., head_h) @ W_O      <- h independent attention
                                                       computations, each free
                                                       to specialize

Encoder layer:  x = LayerNorm(x + SelfAttention(x))
                x = LayerNorm(x + FFN(x))

Decoder layer:  y = LayerNorm(y + MaskedSelfAttention(y))   <- causal mask,
                y = LayerNorm(y + CrossAttention(y, encoder_out))  can't see
                y = LayerNorm(y + FFN(y))                          the future
```

Positional encoding (sinusoidal, many frequencies) is added to the input
embeddings, since attention itself has no notion of word order at all.
Residual connections (`x + Sublayer(x)`, always, unconditionally) plus layer
norm keep gradients stable across many stacked layers — a stronger fix for
the same vanishing-gradient-across-depth problem highway networks solved for
ELMo's character-CNN.

## Files

- `layers.py` — softmax (+ its general backward, not just the NLL shortcut),
  `LayerNorm`, `FeedForward`, sinusoidal positional encoding
- `attention.py` — `MultiHeadAttention`: self-attention, masked
  self-attention, and cross-attention, all through one class
- `encoder_decoder.py` — `EncoderLayer`, `DecoderLayer`: wiring
  attention + FFN with residual connections and layer norm
- `model.py` — `Transformer`: embeddings, stacked encoder/decoder, output
  softmax, full forward/loss/backward/step
- `data.py` — toy sequence-reversal task (standard sanity check for seq2seq
  models — doesn't need a real parallel corpus, but genuinely requires
  correct cross-attention to solve)
- `train.py` — training loop + greedy decoding
- `grad_check_*.py` — numerical gradient checks for every module: layers,
  attention (self/masked/cross, plus a direct check that masked positions
  get exactly zero attention weight), encoder/decoder layers, and the full
  end-to-end model

## Usage

```bash
python3 grad_check_layers.py       # LayerNorm, FeedForward
python3 grad_check_attention.py    # self/masked/cross-attention + masking correctness
python3 grad_check_encdec.py       # EncoderLayer, DecoderLayer
python3 grad_check_model.py        # the WHOLE model, end to end
python3 train.py                   # train on sequence reversal, then decode
```

## Result

Sequence reversal, length-3 sequences, vocab size 8, 100 training examples:
loss drops from 1.45 to 0.0001 over 150 epochs, and the trained model gets
**20/20 exact matches** on held-out test sequences via greedy decoding —
genuine proof the architecture learns to use cross-attention correctly, not
just that the gradients are numerically correct in isolation. A harder
config (length-6 sequences, vocab 15) was tried first and needed
significantly more training than given here to converge — not a bug (every
gradient check still passes), just a harder task needing more epochs/data,
same scaling story as every earlier project's small-corpus caveat.

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS*.
