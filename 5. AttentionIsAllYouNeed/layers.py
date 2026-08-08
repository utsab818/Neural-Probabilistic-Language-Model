import numpy as np


def softmax_rows(x):
    """Row-wise softmax for a 2D or 3D array (softmax over the LAST axis)."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def softmax_backward(p, grad_p):
    """
    General softmax backward (NOT the p-y shortcut, since here softmax's
    output feeds into further computation, not directly into an NLL loss).

    For each row: grad_s = p * (grad_p - sum(grad_p * p))
    p, grad_p: same shape, softmax applied over the last axis.
    """
    dot = np.sum(grad_p * p, axis=-1, keepdims=True)
    return p * (grad_p - dot)


class LayerNorm:
    def __init__(self, dim, eps=1e-6, seed=0):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

    def forward(self, x):
        """x: (T, dim). Normalize each row independently."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        out = self.gamma * x_norm + self.beta
        cache = (x, mean, std, x_norm)
        return out, cache

    def backward(self, cache, grad_out):
        x, mean, std, x_norm = cache
        T, d = x.shape

        grad_gamma = np.sum(grad_out * x_norm, axis=0)
        grad_beta = np.sum(grad_out, axis=0)

        grad_x_norm = grad_out * self.gamma  # (T, d)

        # standard layernorm backward (per-row), derived from
        # x_norm = (x - mean) / std, mean and std both depend on x
        dxnorm_sum = np.sum(grad_x_norm, axis=-1, keepdims=True)
        dxnorm_dot_xnorm = np.sum(grad_x_norm * x_norm, axis=-1, keepdims=True)
        grad_x = (grad_x_norm - dxnorm_sum / d - x_norm * dxnorm_dot_xnorm / d) / std

        return grad_x, {"gamma": grad_gamma, "beta": grad_beta}

    def step(self, grads, lr):
        self.gamma -= lr * grads["gamma"]
        self.beta -= lr * grads["beta"]


class FeedForward:
    def __init__(self, d_model, d_ff, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (d_model, d_ff)) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, 0.1, (d_ff, d_model)) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """x: (T, d_model)"""
        pre1 = x @ self.W1 + self.b1
        h = np.maximum(0, pre1)  # ReLU
        out = h @ self.W2 + self.b2
        cache = (x, pre1, h)
        return out, cache

    def backward(self, cache, grad_out):
        x, pre1, h = cache
        grad_W2 = h.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_h = grad_out @ self.W2.T

        grad_pre1 = grad_h * (pre1 > 0)  # ReLU derivative
        grad_W1 = x.T @ grad_pre1
        grad_b1 = grad_pre1.sum(axis=0)
        grad_x = grad_pre1 @ self.W1.T

        return grad_x, {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2}

    def step(self, grads, lr):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]


def positional_encoding(T, d_model):
    """Sinusoidal positional encoding, shape (T, d_model). No learned params."""
    pe = np.zeros((T, d_model))
    position = np.arange(T)[:, None]
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)
    return pe
