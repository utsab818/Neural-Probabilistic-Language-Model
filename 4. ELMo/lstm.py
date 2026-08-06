import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class LSTMCell:
    def __init__(self, input_dim, hidden_dim, seed=0):
        """
        input_dim: length of x_t (the input at each timestep)
        hidden_dim: length of h_t and c_t

        You need 4 gates, each with its own pair of weight matrices and a
        bias -- recall the equations you derived:

          f_t = sigmoid(W_f @ h_{t-1} + U_f @ x_t + b_f)   forget gate
          i_t = sigmoid(W_i @ h_{t-1} + U_i @ x_t + b_i)   input gate
          c~_t = tanh(W_c @ h_{t-1} + U_c @ x_t + b_c)      candidate
          o_t = sigmoid(W_o @ h_{t-1} + U_o @ x_t + b_o)   output gate

        For each gate, W_* has shape (hidden_dim, hidden_dim) (reads h_{t-1})
        and U_* has shape (hidden_dim, input_dim) (reads x_t), bias is
        (hidden_dim,).

        Store them however you like, but a clean approach: a dict, e.g.
        self.params = {"W_f": ..., "U_f": ..., "b_f": ..., "W_i": ..., ...}
        for all 4 gates (f, i, c, o).

        Initialization: small random for W_*/U_* (use the fan-in scaling
        idea from the NPLM -- divide by sqrt(fan_in), where fan_in is the
        number of columns of that specific matrix). Zero for biases, EXCEPT
        the forget gate's bias b_f -- recall the reasoning from a few turns
        ago: initializing b_f to a positive number (try +1.0) biases the
        network toward "remember by default" at the start of training,
        which helps long-range gradient flow before the network has learned
        anything yet.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        self.params = {}
        for gate in ['f', 'i', 'c', 'o']:
            self.params[f'W_{gate}'] = rng.normal(0, 0.1, (hidden_dim, hidden_dim)) / np.sqrt(hidden_dim)
            self.params[f'U_{gate}'] = rng.normal(0, 0.1, (hidden_dim, input_dim)) / np.sqrt(input_dim)
            self.params[f'b_{gate}'] = np.zeros((hidden_dim,))
        self.params['b_f'] = np.ones((hidden_dim,))

    def forward_step(self, x_t, h_prev, c_prev):
        """
        x_t: (input_dim,)
        h_prev, c_prev: (hidden_dim,)

        Compute f_t, i_t, c~_t, o_t, then c_t, h_t, using the equations
        above. Return h_t, c_t, and a cache of everything backward_step()
        will need (at minimum: x_t, h_prev, c_prev, f_t, i_t, c_tilde_t,
        o_t, c_t, h_t).
        """
        f_t = sigmoid(np.dot(self.params['W_f'], h_prev) + np.dot(self.params['U_f'], x_t) + self.params['b_f'])
        i_t = sigmoid(np.dot(self.params['W_i'], h_prev) + np.dot(self.params['U_i'], x_t) + self.params['b_i'])
        c_tilde_t = np.tanh(np.dot(self.params['W_c'], h_prev) + np.dot(self.params['U_c'], x_t) + self.params['b_c'])
        o_t = sigmoid(np.dot(self.params['W_o'], h_prev) + np.dot(self.params['U_o'], x_t) + self.params['b_o'])
        c_t = f_t * c_prev + i_t * c_tilde_t
        h_t = o_t * np.tanh(c_t)
        cache = (x_t, h_prev, c_prev, f_t, i_t, c_tilde_t, o_t, c_t, h_t)
        return h_t, c_t, cache

    def backward_step(self, cache, grad_h_t, grad_c_t_from_future):
        """
        cache: from forward_step()
        grad_h_t: dL/dh_t, gradient flowing in from wherever h_t was used
                  (could be a loss at this timestep, and/or gradient passed
                  back from timestep t+1's h_{t-1} dependency)
        grad_c_t_from_future: dL/dc_t contribution that already arrived from
                  timestep t+1 (since c_t feeds into c_{t+1} directly via
                  the forget-gate pathway you derived) -- pass in zeros if
                  this is the last timestep.

        You need to derive/implement, step by step:

          1. h_t = o_t * tanh(c_t), so:
             grad_o_t = grad_h_t * tanh(c_t)
             grad_c_t = grad_h_t * o_t * (1 - tanh(c_t)**2) + grad_c_t_from_future
             (the second term of grad_c_t is the NEW contribution flowing in
             through the h_t -> c_t path this timestep; ADD it to what
             already arrived from the future, same "sum contributions from
             every path" rule as always)

          2. c_t = f_t*c_prev + i_t*c_tilde_t, so (all elementwise):
             grad_f_t = grad_c_t * c_prev
             grad_c_prev = grad_c_t * f_t          <- this IS the
                           grad_c_t_from_future you'll pass to the PREVIOUS
                           timestep's backward_step call
             grad_i_t = grad_c_t * c_tilde_t
             grad_c_tilde_t = grad_c_t * i_t

          3. Each gate has a sigmoid or tanh derivative to push through
             before reaching its own W_*/U_*/b_*:
             grad_f_preact = grad_f_t * f_t * (1 - f_t)          (sigmoid')
             grad_i_preact = grad_i_t * i_t * (1 - i_t)          (sigmoid')
             grad_o_preact = grad_o_t * o_t * (1 - o_t)          (sigmoid')
             grad_c_tilde_preact = grad_c_tilde_t * (1 - c_tilde_t**2)  (tanh')

          4. For EACH of the 4 gates (using its own *_preact from step 3),
             same outer-product pattern you've used constantly:
             grad_W_* = outer(*_preact, h_prev)
             grad_U_* = outer(*_preact, x_t)
             grad_b_* = *_preact

          5. grad_h_prev: h_prev feeds into ALL FOUR gates (it's used 4
             times!), so sum the contributions from all 4 gates:
             grad_h_prev = W_f.T @ grad_f_preact + W_i.T @ grad_i_preact
                          + W_c.T @ grad_c_tilde_preact + W_o.T @ grad_o_preact

        Return: (grad_h_prev, grad_c_prev, param_grads) where param_grads is
        a dict with the same keys as self.params.
        """
        x_t, h_prev, c_prev, f_t, i_t, c_tilde_t, o_t, c_t, h_t = cache

        grad_o_t = grad_h_t * np.tanh(c_t)
        grad_c_t = grad_h_t * o_t * (1 - np.tanh(c_t) ** 2) + grad_c_t_from_future

        grad_f_t = grad_c_t * c_prev
        grad_c_prev = grad_c_t * f_t
        grad_i_t = grad_c_t * c_tilde_t
        grad_c_tilde_t = grad_c_t * i_t

        grad_f_preact = grad_f_t * f_t * (1 - f_t)
        grad_i_preact = grad_i_t * i_t * (1 - i_t)
        grad_o_preact = grad_o_t * o_t * (1 - o_t)
        grad_c_tilde_preact = grad_c_tilde_t * (1 - c_tilde_t ** 2)

        param_grads = {}
        param_grads['W_f'] = np.outer(grad_f_preact, h_prev)
        param_grads['U_f'] = np.outer(grad_f_preact, x_t)
        param_grads['b_f'] = grad_f_preact
        param_grads['W_i'] = np.outer(grad_i_preact, h_prev)
        param_grads['U_i'] = np.outer(grad_i_preact, x_t)
        param_grads['b_i'] = grad_i_preact
        param_grads['W_c'] = np.outer(grad_c_tilde_preact, h_prev)
        param_grads['U_c'] = np.outer(grad_c_tilde_preact, x_t)
        param_grads['b_c'] = grad_c_tilde_preact
        param_grads['W_o'] = np.outer(grad_o_preact, h_prev)
        param_grads['U_o'] = np.outer(grad_o_preact, x_t)
        param_grads['b_o'] = grad_o_preact

        grad_h_prev = (self.params['W_f'].T @ grad_f_preact +
                       self.params['W_i'].T @ grad_i_preact +
                       self.params['W_c'].T @ grad_c_tilde_preact +
                       self.params['W_o'].T @ grad_o_preact)

        # grad_x_t: x_t feeds into all 4 gates too, via U_*, same reasoning
        # as grad_h_prev but using U_* matrices instead of W_*
        grad_x_t = (self.params['U_f'].T @ grad_f_preact +
                    self.params['U_i'].T @ grad_i_preact +
                    self.params['U_c'].T @ grad_c_tilde_preact +
                    self.params['U_o'].T @ grad_o_preact)

        return grad_h_prev, grad_c_prev, grad_x_t, param_grads

    def forward_sequence(self, X):
        """
        X: (T, input_dim) -- a whole sequence of inputs, one row per timestep

        Loop forward_step over every timestep, starting h_0 = c_0 = zeros.
        Return: H (T, hidden_dim) -- all hidden states stacked, and a list
        of per-timestep caches (caches[t] = cache from forward_step at
        timestep t), needed by backward_sequence().
        """
        T = X.shape[0]
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        H = np.zeros((T, self.hidden_dim))
        caches = []
        for t in range(T):
            h, c, cache = self.forward_step(X[t], h, c)
            H[t] = h
            caches.append(cache)
        return H, caches

    def backward_sequence(self, caches, grad_H):
        """
        caches: list of per-timestep caches from forward_sequence()
        grad_H: (T, hidden_dim) -- dL/dh_t for EVERY timestep (e.g. if you
                have a loss at every position, like language modeling)

        Loop backward_step over timesteps in REVERSE order (T-1 down to 0).
        At each step t:
          - the grad_h_t fed into backward_step is grad_H[t] PLUS whatever
            grad_h_prev was returned by the (t+1)-th call (since h_t is used
            both for this timestep's loss AND as input to timestep t+1 --
            sum both contributions, same rule as always)
          - the grad_c_t_from_future fed in is whatever grad_c_prev was
            returned by the (t+1)-th call (zeros for the very last timestep)
          - ACCUMULATE each timestep's param_grads into running totals --
            every timestep shares the same W_f/U_f/etc, so their gradients
            must be summed across all T timesteps, not overwritten

        Return: a dict of accumulated param_grads (same keys as self.params,
        each summed over all T timesteps).
        """
        T = len(caches)
        total_param_grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        grad_h_from_future = np.zeros(self.hidden_dim)
        grad_c_from_future = np.zeros(self.hidden_dim)
        grad_X = np.zeros((T, self.input_dim))
        for t in reversed(range(T)):
            grad_h_t = grad_H[t] + grad_h_from_future
            grad_h_from_future, grad_c_from_future, grad_x_t, param_grads = self.backward_step(
                caches[t], grad_h_t, grad_c_from_future)
            grad_X[t] = grad_x_t
            for k in total_param_grads:
                total_param_grads[k] += param_grads[k]
        return total_param_grads, grad_X
