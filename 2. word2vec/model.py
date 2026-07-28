import numpy as np


class Word2Vec:
    def __init__(self, vocab_size, embed_dim, mode="skipgram", seed=0):
        """
        mode: "skipgram" or "cbow" -- determines how `x` is built from
        input_ids in forward(), and how gradients scatter back into C_in
        in backward().
        """
        self.V = vocab_size
        self.m = embed_dim
        self.mode = mode
        rng = np.random.default_rng(seed)

        self.C_in = rng.normal(0, 0.01, size=(vocab_size, embed_dim))
        self.C_out = rng.normal(0, 0.01, size=(vocab_size, embed_dim))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, input_ids, target_ids, negative_ids):
        """
        input_ids: shape depends on self.mode --
          "skipgram": (B,)        one center word per example
          "cbow":     (B, window) several context words per example
        target_ids:   (B,)   the word being predicted (context word for
                      skipgram; the actual center/target word for cbow)
        negative_ids: (B,k)  negative samples for target_ids

        Steps:
          1. compute x:
             - skipgram: x = C_in[input_ids]                    -> (B, m)
             - cbow:     x = mean over axis=1 of C_in[input_ids] -> (B, m)
               (input_ids is (B, window) in this case; look up ALL of them
               at once -- C_in[input_ids] gives (B, window, m) -- then
               average over the window axis)
          2. everything else is identical to what you already built:
             ctx_vecs = C_out[target_ids], score_pos, neg_vecs, score_neg,
             p_pos, p_neg
        """
        if self.mode == "skipgram":
            x = self.C_in[input_ids]
        elif self.mode == "cbow":
            x = np.mean(self.C_in[input_ids], axis=1)
        else:
            raise ValueError(self.mode)

        ctx_vecs = self.C_out[target_ids]
        score_pos = np.sum(x * ctx_vecs, axis=1)
        neg_vecs = self.C_out[negative_ids]
        score_neg = np.sum(x[:, None, :] * neg_vecs, axis=2)
        p_pos = self.sigmoid(score_pos)
        p_neg = self.sigmoid(score_neg)
        cache = (x, ctx_vecs, neg_vecs, p_pos, p_neg)
        return p_pos, p_neg, cache

    def loss(self, p_pos, p_neg):
        """
        p_pos: (B,)   sigmoid(score) for the real pair, per example
        p_neg: (B,k)  sigmoid(score) for each fake pair, per example

        Return: scalar mean negative-sampling loss over the batch:
          L = -log(p_pos) - sum_i(log(1 - p_neg_i))     [per example]
        averaged over the B examples.

        Tip: add a tiny epsilon (1e-12) inside each log to avoid log(0).
        """
        # TODO: implement
        epsilon = 1e-12
        loss_pos = -np.log(p_pos + epsilon)
        loss_neg = -np.sum(np.log(1 - p_neg + epsilon), axis=1)
        loss = np.mean(loss_pos + loss_neg)
        return loss

    def backward(self, cache, input_ids, target_ids, negative_ids):
        """
        cache: (x, ctx_vecs, neg_vecs, p_pos, p_neg) from forward()

        Everything through grad_x is IDENTICAL to skipgram -- only the very
        last step (scattering grad_x back into C_in) depends on self.mode:

          "skipgram": input_ids is (B,) -- one word per example, so
                      grad_C_in[input_ids] += grad_x directly (what you
                      already built).

          "cbow": input_ids is (B, window) -- several words per example,
                  which were AVERAGED to build x. Recall what you derived
                  by hand: if x = average of `window` vectors, the gradient
                  flowing back to each individual contributor is grad_x
                  scaled by 1/window, not the full grad_x.

                  You need every one of the `window` context positions, for
                  every example, to receive grad_x/window, accumulated with
                  += (words can repeat, same as always). Hint: build an
                  array of shape (B, window, m) where every window-slot
                  holds the same (grad_x / window) vector for that example
                  (np.broadcast_to, or a manual tile, both work), then
                  np.add.at(grad_C_in, input_ids, that array) -- np.add.at
                  can scatter using a 2D index array (B, window) into the
                  first axis of grad_C_in, as long as the values array's
                  shape lines up (B, window, m).
        """
        x, ctx_vecs, neg_vecs, p_pos, p_neg = cache
        B = x.shape[0]
        grad_score_pos = (p_pos - 1) / B
        grad_score_neg = p_neg / B
        grad_C_out = np.zeros_like(self.C_out)
        np.add.at(grad_C_out, target_ids, grad_score_pos[:, None] * x)
        np.add.at(grad_C_out, negative_ids, grad_score_neg[:, :, None] * x[:, None, :])
        grad_x = grad_score_pos[:, None] * ctx_vecs + np.sum(grad_score_neg[:, :, None] * neg_vecs, axis=1)

        grad_C_in = np.zeros_like(self.C_in)
        if self.mode == "skipgram":
            np.add.at(grad_C_in, input_ids, grad_x)
        elif self.mode == "cbow":
            window = input_ids.shape[1]
            grad_x_window = grad_x[:, None, :] / window  # shape (B, 1, m)
            grad_x_window = np.broadcast_to(grad_x_window, (B, window, self.m))  # shape (B, window, m)
            np.add.at(grad_C_in, input_ids, grad_x_window)  # scatter into C_in
        else:
            raise ValueError(self.mode)

        return {"C_in": grad_C_in, "C_out": grad_C_out}

    def step(self, grads, lr, weight_decay=1e-4):
        """Plain SGD with L2 weight decay on both embedding tables --
        prevents unbounded embedding-norm growth over long training runs."""
        self.C_in -= lr * (grads["C_in"] + weight_decay * self.C_in)
        self.C_out -= lr * (grads["C_out"] + weight_decay * self.C_out)