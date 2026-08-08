import numpy as np
from encoder_decoder import EncoderLayer, DecoderLayer

rng = np.random.default_rng(0)


def flatten_params(module):
    """Collect (owner_obj, attr_name) pairs for every learnable array in a
    module tree, so we can generically perturb any of them."""
    pairs = []
    for sub in module.__dict__.values():
        if hasattr(sub, "params_list"):
            pass
    return pairs


def check_encoder_layer():
    d_model, num_heads, d_ff, T = 8, 2, 16, 5
    layer = EncoderLayer(d_model, num_heads, d_ff, seed=1)
    x = rng.normal(size=(T, d_model)) * 0.3
    grad_out = rng.normal(size=(T, d_model))

    out, cache = layer.forward(x)
    grad_x, grads = layer.backward(cache, grad_out)

    def loss_now():
        o, _ = layer.forward(x)
        return np.sum(grad_out * o)

    eps = 1e-5
    worst = 0.0

    def check_param(obj, name, analytic):
        nonlocal worst
        param = getattr(obj, name)
        flat_p, flat_a = param.reshape(-1), analytic.reshape(-1)
        idxs = rng.choice(flat_p.size, size=min(3, flat_p.size), replace=False)
        for i in idxs:
            orig = flat_p[i]
            flat_p[i] = orig + eps; lp = loss_now()
            flat_p[i] = orig - eps; lm = loss_now()
            flat_p[i] = orig
            num_grad = (lp - lm) / (2 * eps)
            err = abs(num_grad - flat_a[i]) / max(abs(num_grad), abs(flat_a[i]), 1e-8)
            worst = max(worst, err)

    for name, analytic in grads["attn"].items():
        check_param(layer.self_attn, name, analytic)
    for name, analytic in grads["ln1"].items():
        check_param(layer.ln1, name, analytic)
    for name, analytic in grads["ffn"].items():
        check_param(layer.ffn, name, analytic)
    for name, analytic in grads["ln2"].items():
        check_param(layer.ln2, name, analytic)

    flat_x, flat_gx = x.reshape(-1), grad_x.reshape(-1)
    idxs = rng.choice(flat_x.size, size=min(4, flat_x.size), replace=False)
    for i in idxs:
        orig = flat_x[i]
        flat_x[i] = orig + eps; lp = loss_now()
        flat_x[i] = orig - eps; lm = loss_now()
        flat_x[i] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - flat_gx[i]) / max(abs(num_grad), abs(flat_gx[i]), 1e-8)
        worst = max(worst, err)

    print(f"EncoderLayer: worst relative error = {worst:.2e}")
    return worst


def check_decoder_layer():
    d_model, num_heads, d_ff, T_dec, T_enc = 8, 2, 16, 4, 6
    layer = DecoderLayer(d_model, num_heads, d_ff, seed=2)
    y = rng.normal(size=(T_dec, d_model)) * 0.3
    encoder_out = rng.normal(size=(T_enc, d_model)) * 0.3
    grad_out = rng.normal(size=(T_dec, d_model))

    out, cache = layer.forward(y, encoder_out)
    grad_y, grad_enc, grads = layer.backward(cache, grad_out)

    def loss_now():
        o, _ = layer.forward(y, encoder_out)
        return np.sum(grad_out * o)

    eps = 1e-5
    worst = 0.0

    def check_param(obj, name, analytic):
        nonlocal worst
        param = getattr(obj, name)
        flat_p, flat_a = param.reshape(-1), analytic.reshape(-1)
        idxs = rng.choice(flat_p.size, size=min(3, flat_p.size), replace=False)
        for i in idxs:
            orig = flat_p[i]
            flat_p[i] = orig + eps; lp = loss_now()
            flat_p[i] = orig - eps; lm = loss_now()
            flat_p[i] = orig
            num_grad = (lp - lm) / (2 * eps)
            err = abs(num_grad - flat_a[i]) / max(abs(num_grad), abs(flat_a[i]), 1e-8)
            worst = max(worst, err)

    for section, obj in [("self_attn", layer.self_attn), ("ln1", layer.ln1),
                          ("cross_attn", layer.cross_attn), ("ln2", layer.ln2),
                          ("ffn", layer.ffn), ("ln3", layer.ln3)]:
        for name, analytic in grads[section].items():
            check_param(obj, name, analytic)

    for xname, x, gx in [("y", y, grad_y), ("encoder_out", encoder_out, grad_enc)]:
        flat_x, flat_gx = x.reshape(-1), gx.reshape(-1)
        idxs = rng.choice(flat_x.size, size=min(4, flat_x.size), replace=False)
        for i in idxs:
            orig = flat_x[i]
            flat_x[i] = orig + eps; lp = loss_now()
            flat_x[i] = orig - eps; lm = loss_now()
            flat_x[i] = orig
            num_grad = (lp - lm) / (2 * eps)
            err = abs(num_grad - flat_gx[i]) / max(abs(num_grad), abs(flat_gx[i]), 1e-8)
            worst = max(worst, err)

    print(f"DecoderLayer: worst relative error = {worst:.2e}")
    return worst


if __name__ == "__main__":
    w1 = check_encoder_layer()
    w2 = check_decoder_layer()
    worst = max(w1, w2)
    print("PASS" if worst < 1e-3 else "FAIL", f"(worst={worst:.2e})")
