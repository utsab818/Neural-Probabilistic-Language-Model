import numpy as np


class CharCNN:
    def __init__(self, char_vocab_size, char_embed_dim, filter_width, num_filters, seed=0):
        """
        char_vocab_size: number of distinct characters
        char_embed_dim:  size of each character's embedding vector
        filter_width:    how many adjacent characters each filter looks at
        num_filters:     how many different filters (patterns) to learn

        Parameters:
          char_embeddings: (char_vocab_size, char_embed_dim) -- small random,
                            same reasoning as every other embedding table
          W: (num_filters, filter_width * char_embed_dim) -- one row per
             filter, each row a "weight_vector" for that filter's pattern.
             Scale by 1/sqrt(fan_in), fan_in = filter_width*char_embed_dim
             (same reasoning as the NPLM's H).
          b: (num_filters,) -- zeros
        """
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.filter_width = filter_width
        self.num_filters = num_filters
        rng = np.random.default_rng(seed)

        self.char_embeddings = rng.normal(0, 0.1, (char_vocab_size, char_embed_dim)) / np.sqrt(char_embed_dim)
        self.W = rng.normal(0, 0.1, (num_filters, filter_width * char_embed_dim)) / np.sqrt(filter_width * char_embed_dim)
        self.b = np.zeros((num_filters,))

    def forward(self, char_ids):
        """
        char_ids: (L,) int array -- the character ids making up ONE word
                  (L = word length in characters)

        Steps:
          1. embeds = self.char_embeddings[char_ids]     -> (L, char_embed_dim)
          2. P = L - filter_width + 1                    (number of window
             positions -- if a word is shorter than filter_width, P<=0;
             assume the caller ensures every word is at least filter_width
             characters, e.g. by padding short words)
          3. For each window position p (0 to P-1):
               window = embeds[p : p+filter_width].reshape(-1)   (flatten to
                        length filter_width*char_embed_dim -- this IS the
                        concatenation you reasoned through earlier)
               scores[p] = tanh(self.W @ window + self.b)    -> (num_filters,)
             (a Python loop over p is fine here -- words are short, this
             doesn't need to be fast)
          4. Stack scores into a (P, num_filters) array.
          5. max-pool: for EACH filter (column), find the winning position
             (np.argmax along axis=0) and the max value itself
             (np.max along axis=0) -> output (num_filters,)

        Return: output (num_filters,), and a cache with everything
        backward() will need (char_ids, embeds, all P window scores/tanh
        outputs, and which position won for each filter -- e.g. via
        np.argmax(scores_matrix, axis=0)).
        """
        L = char_ids.shape[0]
        P = L - self.filter_width + 1

        embeds = self.char_embeddings[char_ids]  # (L, char_embed_dim)
        scores_matrix = np.zeros((P, self.num_filters))
        for p in range(P):
            window = embeds[p : p + self.filter_width].reshape(-1)
            scores_matrix[p] = np.tanh(self.W @ window + self.b)
        max_indices = np.argmax(scores_matrix, axis=0)
        output = scores_matrix[max_indices, np.arange(self.num_filters)]
        cache = (char_ids, embeds, scores_matrix, max_indices)
        return output, cache

    def backward(self, cache, grad_output):
        """
        cache: from forward()
        grad_output: (num_filters,) -- dL/d(output), one gradient per filter

        Recall what you derived:
          - max-pooling: gradient flows ONLY to the winning position for
            each filter, zero everywhere else
          - at the winning position: grad_pre_activation = grad_output * (1 - score^2)
            where score is scores_matrix[winning_position, filter]
          - grad_W[filter] = grad_pre_activation[filter] * window_content
            (window_content = embeds[winning_position : winning_position+filter_width]
             .reshape(-1), i.e. the SAME window that won for that filter)
          - grad_b[filter] = grad_pre_activation[filter]
          - grad_window_content = grad_pre_activation[filter] * W[filter]
            -- this needs to be reshaped back to (filter_width, char_embed_dim)
            and scattered (accumulated, np.add.at style) into a
            (L, char_embed_dim) grad_embeds buffer, at the winning window's
            character positions
          - grad_char_embeddings: scatter grad_embeds into the FULL
            (char_vocab_size, char_embed_dim) table, using char_ids
            (np.add.at again, same "shared parameter, repeated use" reasoning
            as every embedding table so far)

        Return: {"char_embeddings": grad_char_embeddings, "W": grad_W, "b": grad_b}

        Note: different filters likely won at DIFFERENT window positions --
        loop over filters (0 to num_filters-1) and handle each one's winning
        position separately.
        """
        char_ids, embeds, scores_matrix, max_indices = cache
        L = char_ids.shape[0]

        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_embeds = np.zeros_like(embeds)
        for f in range(self.num_filters):
            winning_position = max_indices[f]
            score = scores_matrix[winning_position, f]
            grad_pre_activation = grad_output[f] * (1 - score ** 2)
            window_content = embeds[winning_position : winning_position + self.filter_width].reshape(-1)
            grad_W[f] += grad_pre_activation * window_content
            grad_b[f] += grad_pre_activation
            grad_window_content = grad_pre_activation * self.W[f]
            grad_window_content_reshaped = grad_window_content.reshape(self.filter_width, self.char_embed_dim)
            np.add.at(grad_embeds, slice(winning_position, winning_position + self.filter_width), grad_window_content_reshaped)
        grad_char_embeddings = np.zeros_like(self.char_embeddings)
        np.add.at(grad_char_embeddings, char_ids, grad_embeds)
        return {"char_embeddings": grad_char_embeddings, "W": grad_W, "b": grad_b}
