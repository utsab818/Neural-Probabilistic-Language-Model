import numpy as np
from model import BiLM

rng = np.random.default_rng(0)
char_vocab_size, word_vocab_size = 15, 12
char_embed_dim, filter_width, num_filters, hidden_dim = 6, 3, 8, 5
T = 5

model = BiLM(char_vocab_size, word_vocab_size, char_embed_dim, filter_width,
             num_filters, hidden_dim, seed=1)

char_id_seqs = [rng.integers(1, char_vocab_size, size=rng.integers(3, 7)) for _ in range(T)]
word_ids = rng.integers(0, word_vocab_size, size=T)

cache = model.forward(char_id_seqs, word_ids)
loss = model.loss(cache, word_ids)
grads = model.backward(cache, word_ids)
print(f"Initial loss: {loss:.4f}")


def loss_now():
    c = model.forward(char_id_seqs, word_ids)
    return model.loss(c, word_ids)


def check(param, analytic, name, num=5, eps=1e-4):
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
    print(f"{name}: worst relative error = {worst:.2e}")
    return worst


if __name__ == "__main__":
    worst = 0.0
    worst = max(worst, check(model.U, grads["U"], "U"))
    worst = max(worst, check(model.b, grads["b"], "b"))
    for k in model.fwd_lstm.params:
        worst = max(worst, check(model.fwd_lstm.params[k], grads["fwd_lstm"][k], f"fwd_lstm.{k}"))
    for k in model.bwd_lstm.params:
        worst = max(worst, check(model.bwd_lstm.params[k], grads["bwd_lstm"][k], f"bwd_lstm.{k}"))
    for k in model.highway.params:
        worst = max(worst, check(model.highway.params[k], grads["highway"][k], f"highway.{k}"))
    worst = max(worst, check(model.charcnn.char_embeddings, grads["charcnn"]["char_embeddings"], "charcnn.char_embeddings"))
    worst = max(worst, check(model.charcnn.W, grads["charcnn"]["W"], "charcnn.W"))
    worst = max(worst, check(model.charcnn.b, grads["charcnn"]["b"], "charcnn.b"))
    print()
    print("PASS" if worst < 1e-3 else "FAIL", f"(worst={worst:.2e})")
