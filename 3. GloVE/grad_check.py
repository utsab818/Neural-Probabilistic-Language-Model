import numpy as np
from model import GloVe

rng = np.random.default_rng(0)
V, m, B = 30, 6, 8
model = GloVe(vocab_size=V, embed_dim=m, seed=1)

i_ids = rng.integers(0, V, size=B)
j_ids = rng.integers(0, V, size=B)
X_ij = rng.uniform(1, 200, size=B)

diff, f, cache = model.forward(i_ids, j_ids, X_ij)
grads = model.backward(cache)


def loss_now():
    d, ff, _ = model.forward(i_ids, j_ids, X_ij)
    return model.loss(d, ff)


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
    worst = max(check(n) for n in ["W", "W_tilde", "b", "b_tilde"])
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst={worst:.2e})")