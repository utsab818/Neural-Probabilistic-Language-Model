import numpy as np
from model import Transformer

rng = np.random.default_rng(0)

src_vocab, tgt_vocab = 10, 10
d_model, num_heads, d_ff, num_layers = 8, 2, 16, 2
T_src, T_tgt = 5, 4

model = Transformer(src_vocab, tgt_vocab, d_model, num_heads, d_ff, num_layers, max_len=10, seed=1)

src_ids = rng.integers(0, src_vocab, size=T_src)
tgt_input_ids = rng.integers(0, tgt_vocab, size=T_tgt)
tgt_output_ids = rng.integers(0, tgt_vocab, size=T_tgt)

probs, cache = model.forward(src_ids, tgt_input_ids)
loss = model.loss(probs, tgt_output_ids)
grads = model.backward(cache, tgt_output_ids)
print(f"Initial loss: {loss:.4f} (expect ~ln({tgt_vocab})={np.log(tgt_vocab):.4f} at random init)")


def loss_now():
    p, _ = model.forward(src_ids, tgt_input_ids)
    return model.loss(p, tgt_output_ids)


def check(param, analytic, name, num=3, eps=1e-4):
    flat_p, flat_a = param.reshape(-1), analytic.reshape(-1)
    idxs = rng.choice(flat_p.size, size=min(num, flat_p.size), replace=False)
    worst = 0.0
    for i in idxs:
        orig = flat_p[i]
        flat_p[i] = orig + eps; lp = loss_now()
        flat_p[i] = orig - eps; lm = loss_now()
        flat_p[i] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - flat_a[i]) / max(abs(num_grad), abs(flat_a[i]), 1e-8)
        worst = max(worst, err)
    print(f"  {name}: worst relative error = {worst:.2e}")
    return worst


if __name__ == "__main__":
    worst = 0.0
    worst = max(worst, check(model.W_out, grads["W_out"], "W_out"))
    worst = max(worst, check(model.b_out, grads["b_out"], "b_out"))
    worst = max(worst, check(model.src_embed, grads["src_embed"], "src_embed"))
    worst = max(worst, check(model.tgt_embed, grads["tgt_embed"], "tgt_embed"))

    for i, (layer, lgrads) in enumerate(zip(model.encoder_layers, grads["enc_layers"])):
        worst = max(worst, check(layer.self_attn.W_Q, lgrads["attn"]["W_Q"], f"enc[{i}].self_attn.W_Q"))
        worst = max(worst, check(layer.ffn.W1, lgrads["ffn"]["W1"], f"enc[{i}].ffn.W1"))
        worst = max(worst, check(layer.ln1.gamma, lgrads["ln1"]["gamma"], f"enc[{i}].ln1.gamma"))

    for i, (layer, lgrads) in enumerate(zip(model.decoder_layers, grads["dec_layers"])):
        worst = max(worst, check(layer.self_attn.W_Q, lgrads["self_attn"]["W_Q"], f"dec[{i}].self_attn.W_Q"))
        worst = max(worst, check(layer.cross_attn.W_Q, lgrads["cross_attn"]["W_Q"], f"dec[{i}].cross_attn.W_Q"))
        worst = max(worst, check(layer.ffn.W2, lgrads["ffn"]["W2"], f"dec[{i}].ffn.W2"))

    print()
    print("PASS" if worst < 1e-2 else "FAIL", f"(worst={worst:.2e})")
