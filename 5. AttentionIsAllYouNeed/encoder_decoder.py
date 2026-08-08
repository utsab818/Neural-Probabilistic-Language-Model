import numpy as np
from attention import MultiHeadAttention, causal_mask
from layers import LayerNorm, FeedForward


class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, seed=0):
        self.self_attn = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 1)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_cache = self.self_attn.forward(x, x, mask=None)
        x1, ln1_cache = self.ln1.forward(x + attn_out)

        ffn_out, ffn_cache = self.ffn.forward(x1)
        x2, ln2_cache = self.ln2.forward(x1 + ffn_out)

        cache = dict(x=x, attn_cache=attn_cache, ln1_cache=ln1_cache, x1=x1,
                     ffn_cache=ffn_cache, ln2_cache=ln2_cache)
        return x2, cache

    def backward(self, cache, grad_x2):
        grad_x1_plus_ffn, ln2_grads = self.ln2.backward(cache["ln2_cache"], grad_x2)
        # x1 + ffn_out both receive this gradient (residual: sum of paths)
        grad_x1_a = grad_x1_plus_ffn
        grad_ffn_out = grad_x1_plus_ffn

        grad_x1_b, ffn_param_grads = self.ffn.backward(cache["ffn_cache"], grad_ffn_out)
        grad_x1 = grad_x1_a + grad_x1_b

        grad_x_plus_attn, ln1_grads = self.ln1.backward(cache["ln1_cache"], grad_x1)
        grad_x_a = grad_x_plus_attn
        grad_attn_out = grad_x_plus_attn

        grad_x_b, attn_param_grads = self.self_attn.backward(cache["attn_cache"], grad_attn_out)
        grad_x = grad_x_a + grad_x_b

        return grad_x, dict(attn=attn_param_grads, ln1=ln1_grads, ffn=ffn_param_grads, ln2=ln2_grads)

    def step(self, grads, lr):
        self.self_attn.step(grads["attn"], lr)
        self.ln1.step(grads["ln1"], lr)
        self.ffn.step(grads["ffn"], lr)
        self.ln2.step(grads["ln2"], lr)


class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, seed=0):
        self.self_attn = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ln1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, seed=seed + 1)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 2)
        self.ln3 = LayerNorm(d_model)

    def forward(self, y, encoder_out):
        T = y.shape[0]
        mask = causal_mask(T)

        self_attn_out, self_attn_cache = self.self_attn.forward(y, y, mask=mask)
        y1, ln1_cache = self.ln1.forward(y + self_attn_out)

        cross_attn_out, cross_attn_cache = self.cross_attn.forward(y1, encoder_out, mask=None)
        y2, ln2_cache = self.ln2.forward(y1 + cross_attn_out)

        ffn_out, ffn_cache = self.ffn.forward(y2)
        y3, ln3_cache = self.ln3.forward(y2 + ffn_out)

        cache = dict(y=y, encoder_out=encoder_out, self_attn_cache=self_attn_cache,
                     ln1_cache=ln1_cache, y1=y1, cross_attn_cache=cross_attn_cache,
                     ln2_cache=ln2_cache, y2=y2, ffn_cache=ffn_cache, ln3_cache=ln3_cache)
        return y3, cache

    def backward(self, cache, grad_y3):
        grad_y2_plus_ffn, ln3_grads = self.ln3.backward(cache["ln3_cache"], grad_y3)
        grad_y2_a = grad_y2_plus_ffn
        grad_ffn_out = grad_y2_plus_ffn
        grad_y2_b, ffn_param_grads = self.ffn.backward(cache["ffn_cache"], grad_ffn_out)
        grad_y2 = grad_y2_a + grad_y2_b

        grad_y1_plus_cross, ln2_grads = self.ln2.backward(cache["ln2_cache"], grad_y2)
        grad_y1_a = grad_y1_plus_cross
        grad_cross_out = grad_y1_plus_cross
        grad_y1_b, grad_encoder_out, cross_attn_param_grads = self.cross_attn.backward(
            cache["cross_attn_cache"], grad_cross_out)
        grad_y1 = grad_y1_a + grad_y1_b

        grad_y_plus_self, ln1_grads = self.ln1.backward(cache["ln1_cache"], grad_y1)
        grad_y_a = grad_y_plus_self
        grad_self_out = grad_y_plus_self
        grad_y_b, self_attn_param_grads = self.self_attn.backward(cache["self_attn_cache"], grad_self_out)
        grad_y = grad_y_a + grad_y_b

        param_grads = dict(self_attn=self_attn_param_grads, ln1=ln1_grads,
                            cross_attn=cross_attn_param_grads, ln2=ln2_grads,
                            ffn=ffn_param_grads, ln3=ln3_grads)
        return grad_y, grad_encoder_out, param_grads

    def step(self, grads, lr):
        self.self_attn.step(grads["self_attn"], lr)
        self.ln1.step(grads["ln1"], lr)
        self.cross_attn.step(grads["cross_attn"], lr)
        self.ln2.step(grads["ln2"], lr)
        self.ffn.step(grads["ffn"], lr)
        self.ln3.step(grads["ln3"], lr)
