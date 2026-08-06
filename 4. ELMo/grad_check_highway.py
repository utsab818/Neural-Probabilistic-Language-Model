import numpy as np
from highway import Highway

rng = np.random.default_rng(0)
dim = 10
h = Highway(dim=dim, seed=1)
x = rng.normal(size=dim)
grad_y = rng.normal(size=dim)

y, cache = h.forward(x)
grad_x, param_grads = h.backward(cache, grad_y)


def loss_now():
    yy, _ = h.forward(x)
    return np.sum(grad_y * yy)


def check_param(name, num=6, eps=1e-5):
    param = h.params[name]
    analytic = param_grads[name]
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


def check_x(num=6, eps=1e-5):
    worst = 0.0
    idxs = rng.choice(x.size, size=min(num, x.size), replace=False)
    for i in idxs:
        orig = x[i]
        x[i] = orig + eps; lp = loss_now()
        x[i] = orig - eps; lm = loss_now()
        x[i] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - grad_x[i]) / max(abs(num_grad), abs(grad_x[i]), 1e-8)
        worst = max(worst, err)
    print(f"x: worst relative error = {worst:.2e}")
    return worst


if __name__ == "__main__":
    worst = max(check_param(n) for n in h.params)
    worst = max(worst, check_x())
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst={worst:.2e})")
