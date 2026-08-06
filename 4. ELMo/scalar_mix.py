import numpy as np


class ScalarMix:
    """
    Combines several layers' representations into one, using learned,
    task-specific weights:

        w_0, w_1, ... = softmax(s_0, s_1, ...)
        ELMo_word = gamma * sum_j( w_j * h_j )

    s_j and gamma are the only learned parameters here -- everything else
    (the biLM's weights) is meant to be frozen once trained, and a NEW
    ScalarMix gets trained per downstream task.
    """

    def __init__(self, num_layers, seed=0):
        rng = np.random.default_rng(seed)
        self.s = np.zeros(num_layers)  # raw scores, start uniform (softmax(0,0,..)=uniform)
        self.gamma = 1.0

    def softmax(self, s):
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    def forward(self, layer_outputs):
        """
        layer_outputs: list of arrays, all the same shape, one per layer
        Returns: combined output, and a cache for backward()
        """
        w = self.softmax(self.s)
        stacked = np.stack(layer_outputs, axis=0)  # (num_layers, ...)
        weighted_sum = np.tensordot(w, stacked, axes=(0, 0))
        output = self.gamma * weighted_sum
        cache = (w, stacked, weighted_sum)
        return output, cache

    def backward(self, cache, grad_output):
        """
        Returns grad_s (num_layers,), grad_gamma (scalar), and grad_layers
        (list of arrays, one per layer, same shapes as layer_outputs) --
        this last one lets gradient continue flowing back into the biLM
        itself, if you were fine-tuning it (usually frozen in practice, but
        the mechanism should still be correct).
        """
        w, stacked, weighted_sum = cache
        num_layers = len(w)

        grad_gamma = np.sum(grad_output * weighted_sum)
        grad_weighted_sum = grad_output * self.gamma

        # weighted_sum = sum_j(w_j * stacked[j]) -- same dot-product-with-
        # each-side reasoning used throughout: grad w.r.t. w_j is grad dotted
        # with stacked[j]; grad w.r.t. stacked[j] is w_j * grad
        grad_w = np.array([np.sum(grad_weighted_sum * stacked[j]) for j in range(num_layers)])
        grad_layers = [w[j] * grad_weighted_sum for j in range(num_layers)]

        # softmax gradient: same structure as the NPLM's softmax+NLL, but
        # here there's no one-hot target -- it's a generic softmax Jacobian.
        # dw_i/ds_k = w_i*(1[i==k] - w_k), so grad_s_k = sum_i(grad_w_i * dw_i/ds_k)
        grad_s = np.zeros(num_layers)
        for k in range(num_layers):
            grad_s[k] = sum(grad_w[i] * w[i] * ((1.0 if i == k else 0.0) - w[k])
                             for i in range(num_layers))

        return grad_s, grad_gamma, grad_layers

    def step(self, grad_s, grad_gamma, lr):
        self.s -= lr * grad_s
        self.gamma -= lr * grad_gamma
