import numpy as np
from attention import MultiHeadAttention, causal_mask

rng = np.random.default_rng(0)


def check_self_attention(mask=None, label="self-attention"):
    d_model, num_heads, T = 8, 2, 5
    mha = MultiHeadAttention(d_model, num_heads, seed=1)
    X = rng.normal(size=(T, d_model)) * 0.3
    grad_out = rng.normal(size=(T, d_model))

    out, cache = mha.forward(X, X, mask=mask)
    grad_X, param_grads = mha.backward(cache, grad_out)

    def loss_now():
        o, _ = mha.forward(X, X, mask=mask)
        return np.sum(grad_out * o)

    worst = 0.0
    eps = 1e-5
    for name in param_grads:
        param = getattr(mha, name)
        analytic = param_grads[name]
        flat_p, flat_a = param.reshape(-1), analytic.reshape(-1)
        idxs = rng.choice(flat_p.size, size=min(4, flat_p.size), replace=False)
        for i in idxs:
            orig = flat_p[i]
            flat_p[i] = orig + eps; lp = loss_now()
            flat_p[i] = orig - eps; lm = loss_now()
            flat_p[i] = orig
            num_grad = (lp - lm) / (2 * eps)
            err = abs(num_grad - flat_a[i]) / max(abs(num_grad), abs(flat_a[i]), 1e-8)
            worst = max(worst, err)

    flat_x, flat_gx = X.reshape(-1), grad_X.reshape(-1)
    idxs = rng.choice(flat_x.size, size=min(4, flat_x.size), replace=False)
    for i in idxs:
        orig = flat_x[i]
        flat_x[i] = orig + eps; lp = loss_now()
        flat_x[i] = orig - eps; lm = loss_now()
        flat_x[i] = orig
        num_grad = (lp - lm) / (2 * eps)
        err = abs(num_grad - flat_gx[i]) / max(abs(num_grad), abs(flat_gx[i]), 1e-8)
        worst = max(worst, err)

    print(f"{label}: worst relative error = {worst:.2e}")
    return worst


def check_cross_attention():
    d_model, num_heads, T_q, T_k = 8, 2, 4, 6
    mha = MultiHeadAttention(d_model, num_heads, seed=2)
    X_q = rng.normal(size=(T_q, d_model)) * 0.3
    X_kv = rng.normal(size=(T_k, d_model)) * 0.3
    grad_out = rng.normal(size=(T_q, d_model))

    out, cache = mha.forward(X_q, X_kv, mask=None)
    grad_X_q, grad_X_kv, param_grads = mha.backward(cache, grad_out)

    def loss_now():
        o, _ = mha.forward(X_q, X_kv, mask=None)
        return np.sum(grad_out * o)

    worst = 0.0
    eps = 1e-5
    for name in param_grads:
        param = getattr(mha, name)
        analytic = param_grads[name]
        flat_p, flat_a = param.reshape(-1), analytic.reshape(-1)
        idxs = rng.choice(flat_p.size, size=min(4, flat_p.size), replace=False)
        for i in idxs:
            orig = flat_p[i]
            flat_p[i] = orig + eps; lp = loss_now()
            flat_p[i] = orig - eps; lm = loss_now()
            flat_p[i] = orig
            num_grad = (lp - lm) / (2 * eps)
            err = abs(num_grad - flat_a[i]) / max(abs(num_grad), abs(flat_a[i]), 1e-8)
            worst = max(worst, err)

    for xname, x, gx in [("X_q", X_q, grad_X_q), ("X_kv", X_kv, grad_X_kv)]:
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

    print(f"cross-attention: worst relative error = {worst:.2e}")
    return worst


def check_masking_correctness():
    """Verify masked positions genuinely get zero attention weight."""
    d_model, num_heads, T = 8, 2, 5
    mha = MultiHeadAttention(d_model, num_heads, seed=3)
    X = rng.normal(size=(T, d_model))
    mask = causal_mask(T)
    out, cache = mha.forward(X, X, mask=mask)
    aw = cache["attn_weights"]  # (h, T, T)
    # every position t should have ~0 weight on positions > t
    max_future_weight = 0.0
    for t in range(T):
        if t < T - 1:
            max_future_weight = max(max_future_weight, aw[:, t, t + 1:].max())
    print(f"max attention weight on FUTURE positions (should be ~0): {max_future_weight:.2e}")
    return max_future_weight


if __name__ == "__main__":
    w1 = check_self_attention(mask=None, label="self-attention (no mask)")
    T = 5
    w2 = check_self_attention(mask=causal_mask(T), label="masked self-attention")
    w3 = check_cross_attention()
    w4 = check_masking_correctness()

    worst = max(w1, w2, w3)
    print()
    print("PASS" if worst < 1e-3 and w4 < 1e-6 else "FAIL", f"(worst grad err={worst:.2e}, max future weight={w4:.2e})")
