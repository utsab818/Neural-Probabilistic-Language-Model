import numpy as np
from layers import LayerNorm, FeedForward

rng = np.random.default_rng(0)


def check_module(module, x, grad_out, param_names, num=5, eps=1e-5):
    out, cache = module.forward(x)
    grad_x, param_grads = module.backward(cache, grad_out)

    def loss_now():
        o, _ = module.forward(x)
        return np.sum(grad_out * o)

    worst = 0.0
    for name in param_names:
        param = getattr(module, name)
        analytic = param_grads[name]
        flat_p, flat_a = param.reshape(-1), analytic.reshape(-1)
        idxs = rng.choice(flat_p.size, size=min(num, flat_p.size), replace=False)
        for i in idxs:
            orig = flat_p[i]
            flat_p[i] = orig + eps; lp = loss_now()
            flat_p[i] = orig - eps; lm = loss_now()
            flat_p[i] = orig
            num_grad = (lp - lm) / (2 * eps)
            err = abs(num_grad - flat_a[i]) / max(abs(num_grad), abs(flat_a[i]), 1e-8)
            worst = max(worst, err)
        print(f"  {name}: worst rel error so far = {worst:.2e}")

    flat_x, flat_gx = x.reshape(-1), grad_x.reshape(-1)
    idxs = rng.choice(flat_x.size, size=min(num, flat_x.size), replace=False)
    for i in idxs:
        orig = flat_x[i]
        flat_x[i] = orig + eps; lp = loss_now()
        flat_x[i] = orig - eps; lm = loss_now()
        flat_x[i] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - flat_gx[i]) / max(abs(num_grad), abs(flat_gx[i]), 1e-8)
        worst = max(worst, err)
    print(f"  x: worst rel error so far = {worst:.2e}")
    return worst


if __name__ == "__main__":
    T, d, d_ff = 4, 8, 16
    x = rng.normal(size=(T, d))
    grad_out = rng.normal(size=(T, d))

    print("LayerNorm:")
    ln = LayerNorm(d)
    w1 = check_module(ln, x.copy(), grad_out, ["gamma", "beta"])

    print("FeedForward:")
    ff = FeedForward(d, d_ff, seed=1)
    w2 = check_module(ff, x.copy(), grad_out, ["W1", "b1", "W2", "b2"])

    worst = max(w1, w2)
    print("\nPASS" if worst < 1e-3 else "\nFAIL", f"(worst={worst:.2e})")
