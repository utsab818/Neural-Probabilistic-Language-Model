import numpy as np


class GloVe:
    def __init__(self, vocab_size, embed_dim, seed=0):
        """
        Four parameters, all learned:
          W:  "main" word vectors,     shape (V, m)
          W_tilde: "context" vectors,  shape (V, m)  -- separate table, same
                   reasoning as word2vec's C_in/C_out, plus the extra
                   self-dot-product-as-squared-norm argument you worked out
          b:  main word biases,        shape (V,)
          b_tilde: context word biases, shape (V,)

        Small random values for W, W_tilde (same reasoning as word2vec: no
        summing-many-things-together step, so no fan-in scaling needed).
        Biases start at zero, as usual.
        """
        self.V = vocab_size
        self.m = embed_dim
        rng = np.random.default_rng(seed)

        # TODO: implement W, W_tilde, b, b_tilde
        self.W = rng.normal(0, 0.1, (self.V, self.m))
        self.W_tilde = rng.normal(0, 0.1, (self.V, self.m))
        self.b = np.zeros(self.V)
        self.b_tilde = np.zeros(self.V)

    def weight_fn(self, X_ij, x_max=100.0, alpha=0.75):
        """
        f(X_ij) = (X_ij/x_max)^alpha   if X_ij < x_max
                = 1                     otherwise
        X_ij: array of raw co-occurrence counts (any shape)
        Return: array of the same shape, the weight for each entry.
        """
        return np.where(X_ij < x_max, (X_ij / x_max) ** alpha, 1.0)

    def forward(self, i_ids, j_ids, X_ij):
        """
        i_ids, j_ids: (B,) int arrays -- vocabulary indices for the "main"
                      word and "context" word of each pair in this batch
        X_ij: (B,) float array -- the actual raw co-occurrence count for
              each pair (from the cooc dict built in data.py)

        Steps:
          1. w_i = self.W[i_ids]            -> (B, m)
          2. w_j = self.W_tilde[j_ids]       -> (B, m)
          3. dot = row-wise dot product of w_i and w_j -> (B,)
          4. diff = dot + self.b[i_ids] + self.b_tilde[j_ids] - log(X_ij)
          5. f = self.weight_fn(X_ij)

        Return: diff, f, and a cache with whatever backward() needs
        (at minimum: i_ids, j_ids, w_i, w_j, diff, f)
        """
        w_i = self.W[i_ids]
        w_j = self.W_tilde[j_ids]
        dot = np.sum(w_i * w_j, axis=1)
        diff = dot + self.b[i_ids] + self.b_tilde[j_ids] - np.log(X_ij)
        f = self.weight_fn(X_ij)
        cache = (i_ids, j_ids, w_i, w_j, diff, f)
        return diff, f, cache

    def loss(self, diff, f):
        """
        Weighted mean squared error over the batch:
          L = mean( f * diff^2 )
        """
        return np.mean(f * diff ** 2)

    def backward(self, cache):
        """
        cache: (i_ids, j_ids, w_i, w_j, diff, f) from forward()

        Return a dict {"W": grad_W, "W_tilde": grad_W_tilde,
                        "b": grad_b, "b_tilde": grad_b_tilde}

        Recall what you derived by hand:
          grad_diff = 2 * f * diff              (B,)   -- don't forget to
                      divide by batch size B here too, same reasoning as
                      every previous project (loss() used np.mean, so the
                      gradient needs the matching 1/B factor)

          grad_w_i  = grad_diff[:,None] * w_j     (B, m)
          grad_w_j  = grad_diff[:,None] * w_i     (B, m)
          grad_b_i  = grad_diff                   (B,)
          grad_b_j  = grad_diff                   (B,)

        Scatter these into (V,m)/(V,) gradient buffers using np.add.at,
        indexed by i_ids for W/b and j_ids for W_tilde/b_tilde -- same
        repeated-index reasoning as every previous project.
        """
        i_ids, j_ids, w_i, w_j, diff, f = cache
        B = diff.shape[0]

        grad_diff = 2 * f * diff / B

        grad_w_i = grad_diff[:, None] * w_j
        grad_w_j = grad_diff[:, None] * w_i
        grad_b_i = grad_diff
        grad_b_j = grad_diff

        grad_W = np.zeros_like(self.W)
        grad_W_tilde = np.zeros_like(self.W_tilde)
        grad_b = np.zeros_like(self.b)
        grad_b_tilde = np.zeros_like(self.b_tilde)

        np.add.at(grad_W, i_ids, grad_w_i)
        np.add.at(grad_W_tilde, j_ids, grad_w_j)
        np.add.at(grad_b, i_ids, grad_b_i)
        np.add.at(grad_b_tilde, j_ids, grad_b_j)

        return {"W": grad_W, "W_tilde": grad_W_tilde, "b": grad_b, "b_tilde": grad_b_tilde}

    def step(self, grads, lr):
        """Plain SGD update for all four parameters."""
        self.W -= lr * grads["W"]
        self.W_tilde -= lr * grads["W_tilde"]
        self.b -= lr * grads["b"]
        self.b_tilde -= lr * grads["b_tilde"]   