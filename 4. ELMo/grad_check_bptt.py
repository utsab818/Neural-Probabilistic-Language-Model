import numpy as np
from lstm import LSTMCell

rng = np.random.default_rng(0)
input_dim, hidden_dim, T = 5, 4, 6
cell = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, seed=1)

X = rng.normal(size=(T, input_dim)) * 0.5
grad_H = rng.normal(size=(T, hidden_dim))


def virtual_loss():
    H, _ = cell.forward_sequence(X)
    return np.sum(grad_H * H)


H, caches = cell.forward_sequence(X)
param_grads, grad_X = cell.backward_sequence(caches, grad_H)


def check(name, num=6, eps=1e-5):
    param = cell.params[name]
    analytic = param_grads[name]
    flat_p, flat_a = param.reshape(-1), analytic.reshape(-1)
    idxs = rng.choice(flat_p.size, size=min(num, flat_p.size), replace=False)
    worst = 0.0
    for i in idxs:
        orig = flat_p[i]
        flat_p[i] = orig + eps; lp = virtual_loss()
        flat_p[i] = orig - eps; lm = virtual_loss()
        flat_p[i] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - flat_a[i]) / max(abs(num_grad), abs(flat_a[i]), 1e-8)
        worst = max(worst, err)
    print(f"{name}: worst relative error = {worst:.2e}")
    return worst


if __name__ == "__main__":
    worst = max(check(n) for n in cell.params)
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst={worst:.2e})")
