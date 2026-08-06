import numpy as np
from scalar_mix import ScalarMix

rng = np.random.default_rng(0)
num_layers, dim = 2, 6
mix = ScalarMix(num_layers, seed=1)
mix.s = rng.normal(size=num_layers)  # give it nonuniform starting scores for a real check
mix.gamma = 1.3

layer_outputs = [rng.normal(size=dim) for _ in range(num_layers)]
grad_output = rng.normal(size=dim)

output, cache = mix.forward(layer_outputs)
grad_s, grad_gamma, grad_layers = mix.backward(cache, grad_output)


def loss_now():
    out, _ = mix.forward(layer_outputs)
    return np.sum(grad_output * out)


def check_s(eps=1e-5):
    worst = 0.0
    for k in range(num_layers):
        orig = mix.s[k]
        mix.s[k] = orig + eps; lp = loss_now()
        mix.s[k] = orig - eps; lm = loss_now()
        mix.s[k] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - grad_s[k]) / max(abs(num_grad), abs(grad_s[k]), 1e-8)
        worst = max(worst, err)
    print(f"s: worst relative error = {worst:.2e}")
    return worst


def check_gamma(eps=1e-5):
    orig = mix.gamma
    mix.gamma = orig + eps; lp = loss_now()
    mix.gamma = orig - eps; lm = loss_now()
    mix.gamma = orig
    num_grad = (lp - lm) / (2 * eps)
    err = abs(num_grad - grad_gamma) / max(abs(num_grad), abs(grad_gamma), 1e-8)
    print(f"gamma: relative error = {err:.2e}")
    return err


if __name__ == "__main__":
    worst = max(check_s(), check_gamma())
    print("PASS" if worst < 1e-4 else "FAIL", f"(worst={worst:.2e})")
