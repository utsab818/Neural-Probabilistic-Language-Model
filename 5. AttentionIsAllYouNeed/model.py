import numpy as np
from encoder_decoder import EncoderLayer, DecoderLayer
from layers import positional_encoding, softmax_rows


class Transformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len=50, seed=0):
        self.d_model = d_model
        self.max_len = max_len
        rng = np.random.default_rng(seed)

        self.src_embed = rng.normal(0, 0.1, (src_vocab_size, d_model)) / np.sqrt(d_model)
        self.tgt_embed = rng.normal(0, 0.1, (tgt_vocab_size, d_model)) / np.sqrt(d_model)
        self.pe = positional_encoding(max_len, d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, seed=seed + 10 * i)
                                for i in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, seed=seed + 100 + 10 * i)
                                for i in range(num_layers)]

        self.W_out = rng.normal(0, 0.1, (d_model, tgt_vocab_size)) / np.sqrt(d_model)
        self.b_out = np.zeros(tgt_vocab_size)

    def encode(self, src_ids):
        T = len(src_ids)
        x = self.src_embed[src_ids] + self.pe[:T]
        caches = []
        for layer in self.encoder_layers:
            x, cache = layer.forward(x)
            caches.append(cache)
        return x, caches

    def decode(self, tgt_ids, encoder_out):
        T = len(tgt_ids)
        y = self.tgt_embed[tgt_ids] + self.pe[:T]
        caches = []
        for layer in self.decoder_layers:
            y, cache = layer.forward(y, encoder_out)
            caches.append(cache)
        return y, caches

    def forward(self, src_ids, tgt_input_ids):
        """
        src_ids: (T_src,) source token ids
        tgt_input_ids: (T_tgt,) decoder input ids (teacher forcing -- the
                       target sequence shifted right by one, as is standard)
        """
        encoder_out, enc_caches = self.encode(src_ids)
        decoder_out, dec_caches = self.decode(tgt_input_ids, encoder_out)

        logits = decoder_out @ self.W_out + self.b_out  # (T_tgt, tgt_vocab)
        probs = softmax_rows(logits)

        cache = dict(src_ids=src_ids, tgt_input_ids=tgt_input_ids, encoder_out=encoder_out,
                     enc_caches=enc_caches, decoder_out=decoder_out, dec_caches=dec_caches,
                     probs=probs)
        return probs, cache

    def loss(self, probs, tgt_output_ids):
        """tgt_output_ids: (T_tgt,) the TRUE next tokens (target shifted left)."""
        T = len(tgt_output_ids)
        eps = 1e-12
        return -np.mean(np.log(probs[np.arange(T), tgt_output_ids] + eps))

    def backward(self, cache, tgt_output_ids):
        probs = cache["probs"]
        T = len(tgt_output_ids)

        grad_logits = probs.copy()
        grad_logits[np.arange(T), tgt_output_ids] -= 1.0
        grad_logits /= T

        grad_W_out = cache["decoder_out"].T @ grad_logits
        grad_b_out = grad_logits.sum(axis=0)
        grad_decoder_out = grad_logits @ self.W_out.T

        # backprop through decoder stack (reverse order)
        grad_y = grad_decoder_out
        grad_encoder_out_total = np.zeros_like(cache["encoder_out"])
        dec_layer_grads = []
        for layer, dcache in zip(reversed(self.decoder_layers), reversed(cache["dec_caches"])):
            grad_y, grad_enc_contrib, lgrads = layer.backward(dcache, grad_y)
            grad_encoder_out_total += grad_enc_contrib
            dec_layer_grads.append(lgrads)
        dec_layer_grads.reverse()

        # grad_y is now grad w.r.t. (tgt_embed[tgt_input_ids] + pe) -- PE has
        # no params, so this IS the gradient into tgt_embed at those ids
        grad_tgt_embed = np.zeros_like(self.tgt_embed)
        np.add.at(grad_tgt_embed, cache["tgt_input_ids"], grad_y)

        # backprop through encoder stack (reverse order), starting from the
        # accumulated cross-attention gradient
        grad_x = grad_encoder_out_total
        enc_layer_grads = []
        for layer, ecache in zip(reversed(self.encoder_layers), reversed(cache["enc_caches"])):
            grad_x, lgrads = layer.backward(ecache, grad_x)
            enc_layer_grads.append(lgrads)
        enc_layer_grads.reverse()

        grad_src_embed = np.zeros_like(self.src_embed)
        np.add.at(grad_src_embed, cache["src_ids"], grad_x)

        return dict(W_out=grad_W_out, b_out=grad_b_out,
                    src_embed=grad_src_embed, tgt_embed=grad_tgt_embed,
                    enc_layers=enc_layer_grads, dec_layers=dec_layer_grads)

    def step(self, grads, lr):
        self.W_out -= lr * grads["W_out"]
        self.b_out -= lr * grads["b_out"]
        self.src_embed -= lr * grads["src_embed"]
        self.tgt_embed -= lr * grads["tgt_embed"]
        for layer, lgrads in zip(self.encoder_layers, grads["enc_layers"]):
            layer.step(lgrads, lr)
        for layer, lgrads in zip(self.decoder_layers, grads["dec_layers"]):
            layer.step(lgrads, lr)
