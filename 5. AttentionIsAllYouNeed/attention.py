import numpy as np
from layers import softmax_rows, softmax_backward


def causal_mask(T):
    """(T, T) mask: 0 where allowed, -1e9 where forbidden (future positions)."""
    mask = np.triu(np.ones((T, T)), k=1) * -1e9
    return mask


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.normal(0, 0.1, (d_model, d_model)) * scale
        self.b_Q = np.zeros(d_model)
        self.W_K = rng.normal(0, 0.1, (d_model, d_model)) * scale
        self.b_K = np.zeros(d_model)
        self.W_V = rng.normal(0, 0.1, (d_model, d_model)) * scale
        self.b_V = np.zeros(d_model)
        self.W_O = rng.normal(0, 0.1, (d_model, d_model)) * scale
        self.b_O = np.zeros(d_model)

    def _split_heads(self, X):
        """(T, d_model) -> (h, T, d_k)"""
        T = X.shape[0]
        return X.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)

    def _merge_heads(self, X):
        """(h, T, d_k) -> (T, d_model)"""
        h, T, d_k = X.shape
        return X.transpose(1, 0, 2).reshape(T, h * d_k)

    def forward(self, X_q, X_kv, mask=None):
        """
        X_q:  (T_q, d_model) -- query source (decoder state for cross-attn,
              same as X_kv for self-attention)
        X_kv: (T_k, d_model) -- key/value source
        mask: (T_q, T_k) or None -- additive mask (0 = allowed, -1e9 = forbidden)
        """
        Q_all = X_q @ self.W_Q + self.b_Q
        K_all = X_kv @ self.W_K + self.b_K
        V_all = X_kv @ self.W_V + self.b_V

        Q = self._split_heads(Q_all)  # (h, T_q, d_k)
        K = self._split_heads(K_all)  # (h, T_k, d_k)
        V = self._split_heads(V_all)  # (h, T_k, d_k)

        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)  # (h, T_q, T_k)
        if mask is not None:
            scores = scores + mask[None, :, :]

        attn_weights = softmax_rows(scores)  # (h, T_q, T_k)
        head_out = attn_weights @ V          # (h, T_q, d_k)

        concat = self._merge_heads(head_out)  # (T_q, d_model)
        out = concat @ self.W_O + self.b_O

        cache = dict(X_q=X_q, X_kv=X_kv, Q_all=Q_all, K_all=K_all, V_all=V_all,
                     Q=Q, K=K, V=V, scores=scores, attn_weights=attn_weights,
                     head_out=head_out, concat=concat, mask=mask, self_attn=(X_q is X_kv))
        return out, cache

    def backward(self, cache, grad_out):
        X_q, X_kv = cache["X_q"], cache["X_kv"]
        Q, K, V = cache["Q"], cache["K"], cache["V"]
        attn_weights, concat = cache["attn_weights"], cache["concat"]

        grad_W_O = concat.T @ grad_out
        grad_b_O = grad_out.sum(axis=0)
        grad_concat = grad_out @ self.W_O.T

        grad_head_out = self._split_heads(grad_concat)  # (h, T_q, d_k)

        grad_attn_weights = grad_head_out @ V.transpose(0, 2, 1)  # (h, T_q, T_k)
        grad_V = attn_weights.transpose(0, 2, 1) @ grad_head_out  # (h, T_k, d_k)

        grad_scores = softmax_backward(attn_weights, grad_attn_weights)  # (h,T_q,T_k)
        grad_scores = grad_scores / np.sqrt(self.d_k)

        grad_Q = grad_scores @ K            # (h, T_q, d_k)
        grad_K = grad_scores.transpose(0, 2, 1) @ Q  # (h, T_k, d_k)

        grad_Q_all = self._merge_heads(grad_Q)
        grad_K_all = self._merge_heads(grad_K)
        grad_V_all = self._merge_heads(grad_V)

        grad_W_Q = X_q.T @ grad_Q_all
        grad_b_Q = grad_Q_all.sum(axis=0)
        grad_X_q = grad_Q_all @ self.W_Q.T

        grad_W_K = X_kv.T @ grad_K_all
        grad_b_K = grad_K_all.sum(axis=0)
        grad_X_kv_from_K = grad_K_all @ self.W_K.T

        grad_W_V = X_kv.T @ grad_V_all
        grad_b_V = grad_V_all.sum(axis=0)
        grad_X_kv_from_V = grad_V_all @ self.W_V.T

        grad_X_kv = grad_X_kv_from_K + grad_X_kv_from_V

        param_grads = {"W_Q": grad_W_Q, "b_Q": grad_b_Q, "W_K": grad_W_K, "b_K": grad_b_K,
                        "W_V": grad_W_V, "b_V": grad_b_V, "W_O": grad_W_O, "b_O": grad_b_O}

        if cache["self_attn"]:
            # X_q and X_kv were the SAME tensor -- sum both contributions
            grad_X = grad_X_q + grad_X_kv
            return grad_X, param_grads
        else:
            return grad_X_q, grad_X_kv, param_grads

    def step(self, param_grads, lr):
        for name in ["W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O", "b_O"]:
            setattr(self, name, getattr(self, name) - lr * param_grads[name])
