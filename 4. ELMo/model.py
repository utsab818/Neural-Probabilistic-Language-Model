import numpy as np
from lstm import LSTMCell
from charcnn import CharCNN
from highway import Highway


class BiLM:
    """
    A bidirectional language model, following ELMo's architecture:
      word -> CharCNN -> Highway -> shared word representation x_t
      x_t sequence -> forward LSTM  -> H_fwd  (predicts word_{t+1})
      x_t sequence -> backward LSTM -> H_bwd  (predicts word_{t-1})
      tied output projection U, b used for BOTH directions

    CharCNN and Highway are shared (tied) between directions -- only the
    two LSTMs have separate weights.
    """

    def __init__(self, char_vocab_size, word_vocab_size, char_embed_dim,
                 filter_width, num_filters, hidden_dim, seed=0):
        self.word_vocab_size = word_vocab_size
        self.repr_dim = num_filters  # highway preserves dim, CharCNN output = num_filters

        self.charcnn = CharCNN(char_vocab_size, char_embed_dim, filter_width, num_filters, seed=seed)
        self.highway = Highway(self.repr_dim, seed=seed + 1)
        self.fwd_lstm = LSTMCell(self.repr_dim, hidden_dim, seed=seed + 2)
        self.bwd_lstm = LSTMCell(self.repr_dim, hidden_dim, seed=seed + 3)

        rng = np.random.default_rng(seed + 4)
        self.U = rng.normal(0, 0.1, (word_vocab_size, hidden_dim)) / np.sqrt(hidden_dim)
        self.b = np.zeros(word_vocab_size)

    def softmax(self, o):
        o = o - o.max()
        e = np.exp(o)
        return e / e.sum()

    def forward(self, char_id_seqs, word_ids):
        """
        char_id_seqs: list of T arrays, one per word (variable length each)
        word_ids: (T,) int array, true word ids for computing the LM loss

        Returns everything needed for loss() and backward().
        """
        T = len(char_id_seqs)

        # 1. CharCNN + Highway -> shared word representations X (T, repr_dim)
        X = np.zeros((T, self.repr_dim))
        cnn_caches, hw_caches = [], []
        for t in range(T):
            cnn_out, cnn_cache = self.charcnn.forward(char_id_seqs[t])
            hw_out, hw_cache = self.highway.forward(cnn_out)
            X[t] = hw_out
            cnn_caches.append(cnn_cache)
            hw_caches.append(hw_cache)

        # 2. forward LSTM over X in normal order
        H_fwd, fwd_caches = self.fwd_lstm.forward_sequence(X)

        # 3. backward LSTM over X reversed, then flip results back to normal order
        X_rev = X[::-1]
        H_bwd_rev, bwd_caches = self.bwd_lstm.forward_sequence(X_rev)
        H_bwd = H_bwd_rev[::-1]

        # 4. predictions: forward LSTM at position t predicts word_ids[t+1]
        #    (only valid for t = 0..T-2); backward LSTM at position t
        #    predicts word_ids[t-1] (only valid for t = 1..T-1)
        fwd_probs = []
        for t in range(T - 1):
            logits = self.U @ H_fwd[t] + self.b
            fwd_probs.append(self.softmax(logits))

        bwd_probs = []
        for t in range(1, T):
            logits = self.U @ H_bwd[t] + self.b
            bwd_probs.append(self.softmax(logits))

        cache = dict(X=X, cnn_caches=cnn_caches, hw_caches=hw_caches,
                     H_fwd=H_fwd, fwd_caches=fwd_caches,
                     H_bwd=H_bwd, bwd_caches=bwd_caches,
                     fwd_probs=fwd_probs, bwd_probs=bwd_probs, T=T)
        return cache

    def loss(self, cache, word_ids):
        T = cache["T"]
        fwd_targets = word_ids[1:T]
        bwd_targets = word_ids[0:T - 1]
        eps = 1e-12
        fwd_loss = np.mean([-np.log(p[tgt] + eps) for p, tgt in zip(cache["fwd_probs"], fwd_targets)])
        bwd_loss = np.mean([-np.log(p[tgt] + eps) for p, tgt in zip(cache["bwd_probs"], bwd_targets)])
        return fwd_loss + bwd_loss

    def backward(self, cache, word_ids):
        T = cache["T"]
        fwd_targets = word_ids[1:T]
        bwd_targets = word_ids[0:T - 1]
        n_fwd, n_bwd = len(fwd_targets), len(bwd_targets)

        grad_U = np.zeros_like(self.U)
        grad_b = np.zeros_like(self.b)
        grad_H_fwd = np.zeros((T, self.fwd_lstm.hidden_dim))
        grad_H_bwd = np.zeros((T, self.bwd_lstm.hidden_dim))

        for t in range(T - 1):
            p = cache["fwd_probs"][t]
            grad_o = p.copy()
            grad_o[fwd_targets[t]] -= 1.0
            grad_o /= n_fwd
            grad_U += np.outer(grad_o, cache["H_fwd"][t])
            grad_b += grad_o
            grad_H_fwd[t] += self.U.T @ grad_o

        for idx, t in enumerate(range(1, T)):
            p = cache["bwd_probs"][idx]
            grad_o = p.copy()
            grad_o[bwd_targets[idx]] -= 1.0
            grad_o /= n_bwd
            grad_U += np.outer(grad_o, cache["H_bwd"][t])
            grad_b += grad_o
            grad_H_bwd[t] += self.U.T @ grad_o

        # backprop through the LSTMs
        fwd_param_grads, grad_X_fwd = self.fwd_lstm.backward_sequence(cache["fwd_caches"], grad_H_fwd)
        # backward LSTM was run on X_rev, so grad_H must be reversed to match
        grad_H_bwd_rev = grad_H_bwd[::-1]
        bwd_param_grads, grad_X_bwd_rev = self.bwd_lstm.backward_sequence(cache["bwd_caches"], grad_H_bwd_rev)
        grad_X_bwd = grad_X_bwd_rev[::-1]

        # X is shared (tied) between both directions, so their gradients sum
        grad_X = grad_X_fwd + grad_X_bwd

        # backprop through Highway and CharCNN for each word, accumulating
        # (shared parameters across all words in the sentence)
        hw_param_grads = {k: np.zeros_like(v) for k, v in self.highway.params.items()}
        cnn_param_grads = {"char_embeddings": np.zeros_like(self.charcnn.char_embeddings),
                            "W": np.zeros_like(self.charcnn.W),
                            "b": np.zeros_like(self.charcnn.b)}
        for t in range(T):
            grad_hw_out = grad_X[t]
            grad_cnn_out, hw_g = self.highway.backward(cache["hw_caches"][t], grad_hw_out)
            for k in hw_param_grads:
                hw_param_grads[k] += hw_g[k]
            cnn_g = self.charcnn.backward(cache["cnn_caches"][t], grad_cnn_out)
            for k in cnn_param_grads:
                cnn_param_grads[k] += cnn_g[k]

        return {
            "U": grad_U, "b": grad_b,
            "fwd_lstm": fwd_param_grads, "bwd_lstm": bwd_param_grads,
            "highway": hw_param_grads, "charcnn": cnn_param_grads,
        }

    def step(self, grads, lr):
        self.U -= lr * grads["U"]
        self.b -= lr * grads["b"]
        for k in self.fwd_lstm.params:
            self.fwd_lstm.params[k] -= lr * grads["fwd_lstm"][k]
        for k in self.bwd_lstm.params:
            self.bwd_lstm.params[k] -= lr * grads["bwd_lstm"][k]
        for k in self.highway.params:
            self.highway.params[k] -= lr * grads["highway"][k]
        self.charcnn.char_embeddings -= lr * grads["charcnn"]["char_embeddings"]
        self.charcnn.W -= lr * grads["charcnn"]["W"]
        self.charcnn.b -= lr * grads["charcnn"]["b"]
