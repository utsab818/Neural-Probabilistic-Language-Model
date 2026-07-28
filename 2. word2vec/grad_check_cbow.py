import numpy as np
from model import Word2Vec

rng = np.random.default_rng(0)
V, m, B, k, window = 30, 6, 5, 4, 4
model = Word2Vec(vocab_size=V, embed_dim=m, mode="cbow", seed=1)

input_ids = rng.integers(0, V, size=(B, window))   # context words
target_ids = rng.integers(0, V, size=B)              # the word being predicted
negative_ids = rng.integers(0, V, size=(B, k))

p_pos, p_neg, cache = model.forward(input_ids, target_ids, negative_ids)
grads = model.backward(cache, input_ids, target_ids, negative_ids)


def loss_now():
    pp, pn, _ = model.forward(input_ids, target_ids, negative_ids)
    return model.loss(pp, pn)


def check(name, num=8, eps=1e-5):
    param = getattr(model, name)
    analytic = grads[name]
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
    worst = max(check(n) for n in ["C_in", "C_out"])
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst={worst:.2e})")
