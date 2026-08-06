import numpy as np
from charcnn import CharCNN

rng = np.random.default_rng(0)
V, m, k, n = 20, 6, 3, 8
cnn = CharCNN(char_vocab_size=V, char_embed_dim=m, filter_width=k, num_filters=n, seed=1)

char_ids = rng.integers(0, V, size=9)
grad_output = rng.normal(size=n)

output, cache = cnn.forward(char_ids)
grads = cnn.backward(cache, grad_output)


def loss_now():
    out, _ = cnn.forward(char_ids)
    return np.sum(grad_output * out)


def check(name, num=6, eps=1e-5):
    param = getattr(cnn, name)
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
    worst = max(check(n) for n in ["char_embeddings", "W", "b"])
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst={worst:.2e})")
