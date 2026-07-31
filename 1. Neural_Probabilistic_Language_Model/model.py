import numpy as np


class NPLM:
    def __init__(self, vocab_size, context_size, embed_dim, hidden_dim, seed=0):
        """
        vocab_size (V): number of words in the vocabulary
        context_size (n-1): number of previous words used as context
        embed_dim (m): size of each word's embedding vector
        hidden_dim (h): number of hidden units

        You need to initialize 5 things, storing them as self.C, self.H,
        self.d, self.U, self.b:

          C: embedding table, shape (V, m) -- small random values, no fan-in scaling
          H: context->hidden weights, shape (h, (n-1)*m) -- random / sqrt(fan_in),
             where fan_in = (n-1)*m
          d: hidden bias, shape (h,) -- zeros
          U: hidden->output weights, shape (V, h) -- random / sqrt(fan_in),
             where fan_in = h
          b: output bias, shape (V,) -- zeros

        Store V, context_size, embed_dim, hidden_dim on self too -- you'll need
        them later in forward()/backward() for reshaping.
        """
        self.V = vocab_size
        self.n1 = context_size
        self.m = embed_dim
        self.h = hidden_dim
        rng = np.random.default_rng(seed)

        self.C = rng.normal(0, 0.01, (vocab_size, embed_dim))
        self.H = rng.normal(0, 1 / np.sqrt((context_size) * embed_dim), (hidden_dim, context_size * embed_dim))
        self.d = np.zeros(hidden_dim)
        self.U = rng.normal(0, 1 / np.sqrt(hidden_dim), (vocab_size, hidden_dim))
        self.b = np.zeros(vocab_size)

    def forward(self, context_ids):
        """
        context_ids: int array of shape (B, n-1) -- B examples, each with n-1
                     context word indices.

        Returns:
          probs: (B, V) softmax probabilities
          cache: whatever intermediate values you'll need in backward() --
                 at minimum you'll want context_ids, x, z, and probs.

        Steps to implement (batched, using self.C, self.H, self.d, self.U, self.b):
          1. embeds = look up self.C at context_ids               -> (B, n-1, m)
          2. x = reshape embeds into (B, (n-1)*m)                  -> concatenation
          3. a = x @ H.T + d                                        -> (B, h)
          4. z = tanh(a)                                            -> (B, h)
          5. o = z @ U.T + b                                        -> (B, V)
          6. probs = softmax(o), row-wise, numerically stable
             (subtract each row's max before exponentiating)
        """
        B = context_ids.shape[0]
        embeds = self.C[context_ids]
        x = embeds.reshape(B, -1)
        a = x @ self.H.T + self.d
        z = np.tanh(a)
        o = z @ self.U.T + self.b
        o_max = np.max(o, axis=1, keepdims=True)
        exp_o = np.exp(o - o_max)
        probs = exp_o / np.sum(exp_o, axis=1, keepdims=True)
        cache = (context_ids, x, z, probs)
        return probs, cache

    def loss(self, probs, targets):
        """
        probs: (B, V) softmax output from forward()
        targets: (B,) int array of correct next-word indices

        Return: a single scalar -- the mean negative log-likelihood across
        the batch, i.e. mean_over_batch( -log(probs[i, targets[i]]) ).

        Tip: add a tiny epsilon like 1e-12 inside the log to avoid log(0)
        in case a probability underflows to exactly 0.0.
        """
        B = probs.shape[0]
        epsilon = 1e-12
        log_probs = np.log(probs[np.arange(B), targets] + epsilon)
        mean_nll = -np.mean(log_probs)
        return mean_nll

    def backward(self, cache, targets):
        """
        cache: (context_ids, x, z, probs) from forward()
        targets: (B,) int array of correct next-word indices

        Return a dict with keys "C", "H", "d", "U", "b" -- gradients with the
        SAME shapes as the corresponding parameters, averaged over the batch.

        Recall the chain you derived:
          grad_o = (probs - one_hot(targets)) / B      <- divide by B here!
          grad_U = grad_o.T @ z
          grad_b = grad_o.sum(axis=0)
          grad_z = grad_o @ U
          grad_a = grad_z * (1 - z**2)
          grad_H = grad_a.T @ x
          grad_d = grad_a.sum(axis=0)
          grad_x = grad_a @ H
          grad_C = scatter-add grad_x (reshaped to (B, n-1, m)) into a
                   zeros_like(C) buffer, indexed by context_ids
                   (hint: np.add.at is the tool for this)
        """
        context_ids, x, z, probs = cache
        B = probs.shape[0]
        grad_o = (probs - np.eye(self.V)[targets]) / B
        grad_U = grad_o.T @ z
        grad_b = np.sum(grad_o, axis=0)
        grad_z = grad_o @ self.U
        grad_a = grad_z * (1 - z ** 2)
        grad_H = grad_a.T @ x
        grad_d = np.sum(grad_a, axis=0)
        grad_x = grad_a @ self.H
        grad_C = np.zeros_like(self.C)
        grad_x_reshaped = grad_x.reshape(B, self.n1, self.m)
        np.add.at(grad_C, context_ids, grad_x_reshaped)
        return {"C": grad_C, "H": grad_H, "d": grad_d, "U": grad_U, "b": grad_b}

    def step(self, grads, lr, weight_decay=1e-4):
        """
        Plain SGD update: param -= lr * grad, for all 5 parameters.
        Optionally add L2 weight decay (param -= lr*weight_decay*param) to
        H and U only (not biases, not embeddings -- common convention).
        """

        self.C -= lr * grads["C"]
        self.H -= lr * (grads["H"] + weight_decay * self.H)
        self.d -= lr * grads["d"]
        self.U -= lr * (grads["U"] + weight_decay * self.U)
        self.b -= lr * grads["b"]