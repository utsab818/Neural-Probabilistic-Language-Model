import numpy as np
from lstm import LSTMCell

rng = np.random.default_rng(0)
input_dim, hidden_dim = 6, 5
cell = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, seed=1)

x_t = rng.normal(size=input_dim)
h_prev = rng.normal(size=hidden_dim) * 0.1
c_prev = rng.normal(size=hidden_dim) * 0.1
grad_h_t = rng.normal(size=hidden_dim)
grad_c_t_from_future = rng.normal(size=hidden_dim)


def virtual_loss():
    h_t, c_t, _ = cell.forward_step(x_t, h_prev, c_prev)
    return np.sum(grad_h_t * h_t) + np.sum(grad_c_t_from_future * c_t)


h_t, c_t, cache = cell.forward_step(x_t, h_prev, c_prev)
grad_h_prev, grad_c_prev, grad_x_t, param_grads = cell.backward_step(cache, grad_h_t, grad_c_t_from_future)


def check_param(name, num=6, eps=1e-5):
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


def check_input(name, vec, analytic, num=6, eps=1e-5):
    worst = 0.0
    idxs = rng.choice(vec.size, size=min(num, vec.size), replace=False)
    for i in idxs:
        orig = vec[i]
        vec[i] = orig + eps; lp = virtual_loss()
        vec[i] = orig - eps; lm = virtual_loss()
        vec[i] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - analytic[i]) / max(abs(num_grad), abs(analytic[i]), 1e-8)
        worst = max(worst, err)
    print(f"{name}: worst relative error = {worst:.2e}")
    return worst


if __name__ == "__main__":
    worst = 0.0
    for name in cell.params:
        worst = max(worst, check_param(name))
    worst = max(worst, check_input("h_prev", h_prev, grad_h_prev))
    worst = max(worst, check_input("c_prev", c_prev, grad_c_prev))
    worst = max(worst, check_input("x_t", x_t, grad_x_t))
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst={worst:.2e})")
