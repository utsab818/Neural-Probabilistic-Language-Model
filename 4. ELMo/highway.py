import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Highway:
    def __init__(self, dim, seed=0):
        """
        dim: length of the input/output vector (highway layers preserve
             dimensionality -- x and y are the same size, since y is a
             MIXTURE of x and a transformed version of x)

        Parameters (recall the equations you derived):
          g = sigmoid(W_g @ x + b_g)      transform gate
          T = tanh(W_t @ x + b_t)          the "fully transformed" version
          y = (1-g)*x + g*T

        W_g, W_t: (dim, dim), scaled by 1/sqrt(dim) (fan-in)
        b_g, b_t: (dim,), zeros
        """
        self.dim = dim
        rng = np.random.default_rng(seed)

        self.params = {}
        self.params['W_g'] = rng.normal(0, 0.1, (dim, dim)) / np.sqrt(dim)
        self.params['b_g'] = np.zeros((dim,))
        self.params['W_t'] = rng.normal(0, 0.1, (dim, dim)) / np.sqrt(dim)
        self.params['b_t'] = np.zeros((dim,))

    def forward(self, x):
        """
        x: (dim,)
        Compute g, T, y as above. Return y and a cache with everything
        backward() will need (x, g, T, y).
        """

        g = sigmoid(np.dot(self.params['W_g'], x) + self.params['b_g'])
        T = np.tanh(np.dot(self.params['W_t'], x) + self.params['b_t'])
        y = (1 - g) * x + g * T
        cache = (x, g, T, y)
        return y, cache

    def backward(self, cache, grad_y):
        """
        cache: (x, g, T, y) from forward()
        grad_y: (dim,) -- dL/dy

        Derive this yourself, using the exact same building blocks as
        everywhere else in this project:

          y = (1-g)*x + g*T     (elementwise)

          Treat this like the LSTM's gate equations: y depends on g, x, AND
          T, so you need grad_g, grad_x (from THIS equation specifically --
          there's also a second path into grad_x you'll add next), and
          grad_T, each via ordinary product-rule reasoning (if y = a*b,
          dy/da = b and dy/db = a, applied per elementwise term above).

          Then push grad_g back through sigmoid (g*(1-g)) to its
          pre-activation, and grad_T back through tanh (1-T^2) to ITS
          pre-activation -- same pattern as the LSTM's gates.

          Each pre-activation gradient then gives you grad_W_*, grad_b_*
          (outer product / direct assignment, same as always), AND a second
          contribution to grad_x (since x also feeds INTO the g and T
          computations, not just the outer y=(1-g)*x+g*T equation) -- sum
          ALL of x's contributions together (the y=(1-g)*x term directly,
          PLUS the two indirect paths through g's and T's pre-activations).

        Return: (grad_x, param_grads) where param_grads has keys
        "W_g", "b_g", "W_t", "b_t".
        """
        x, g, T, y = cache
        grad_g = grad_y * (T - x)
        grad_x_direct = grad_y * (1 - g)
        grad_T = grad_y * g

        grad_g_preact = grad_g * g * (1 - g)
        grad_T_preact = grad_T * (1 - T ** 2)

        grad_W_g = np.outer(grad_g_preact, x)
        grad_b_g = grad_g_preact
        grad_W_t = np.outer(grad_T_preact, x)
        grad_b_t = grad_T_preact

        grad_x = grad_x_direct + self.params['W_g'].T @ grad_g_preact + self.params['W_t'].T @ grad_T_preact

        param_grads = {'W_g': grad_W_g, 'b_g': grad_b_g, 'W_t': grad_W_t, 'b_t': grad_b_t}
        return grad_x, param_grads
