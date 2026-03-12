"""
Alternative world models for comparison with Dreamer's RSSM.

Implements two lightweight models that share the same interface as RSSM:
  - LinearSSM  : next_obs = A @ obs + B @ action  (least-squares fit, numpy)
  - MLPDynamics: next_obs = MLP(concat(obs, action))  (online SGD)

Both models operate on flat state observations (obs_dim,) rather than image
embeddings, and expose the same methods used by the Dreamer training loop:

    initial(batch_size)            -> state_dict
    observe(obs, action, state)    -> (post, prior)
    obs_step(state, action, obs)   -> (post, prior)
    img_step(state, action)        -> state
    get_feat(state)                -> flat feature tensor
    get_dist(state)                -> tfp distribution (for KL logging)
    imagine(action, state)         -> prior sequence

State dict for both models: {'obs': Tensor[batch, obs_dim]}
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras import mixed_precision as prec

import tools


class LinearSSM(tools.Module):
    """
    Linear state-space model fitted by least squares.

    Dynamics: next_obs ≈ obs @ A  +  action @ B
    where  A ∈ R^{obs_dim × obs_dim},  B ∈ R^{action_dim × obs_dim}.

    A and B are stored as non-trainable tf.Variables so they are included
    in the module's checkpoint and are accessible from within @tf.functions.
    They are updated by calling _refit() which solves the ordinary least-squares
    problem on the accumulated transition buffer.
    """

    def __init__(self, obs_dim, action_dim, refit_every=1000, max_buffer=50_000):
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._refit_every = refit_every
        self._max_buffer = max_buffer
        self._step_count = 0

        # Non-trainable TF variables used inside @tf.function.
        self._A_var = tf.Variable(
            tf.zeros([obs_dim, obs_dim]), trainable=False,
            dtype=tf.float32, name='linear_A')
        self._B_var = tf.Variable(
            tf.zeros([action_dim, obs_dim]), trainable=False,
            dtype=tf.float32, name='linear_B')

        # Numpy transition buffer for least-squares fitting.
        self._obs_buf = []
        self._act_buf = []
        self._next_obs_buf = []

    # ------------------------------------------------------------------
    # Buffer management (called from Python / tf.py_function)
    # ------------------------------------------------------------------

    def update_buffer(self, state_batch, action_batch):
        """Add transitions from a training batch and refit if due.

        Args:
            state_batch : numpy array  [B, T, obs_dim]
            action_batch: numpy array  [B, T, action_dim]
        """
        B, T = state_batch.shape[:2]
        for b in range(B):
            for t in range(T - 1):
                self._obs_buf.append(state_batch[b, t].astype(np.float64))
                self._act_buf.append(action_batch[b, t].astype(np.float64))
                self._next_obs_buf.append(state_batch[b, t + 1].astype(np.float64))

        # Cap buffer size to avoid memory growth.
        if len(self._obs_buf) > self._max_buffer:
            self._obs_buf = self._obs_buf[-self._max_buffer:]
            self._act_buf = self._act_buf[-self._max_buffer:]
            self._next_obs_buf = self._next_obs_buf[-self._max_buffer:]

        self._step_count += B * T
        if self._step_count >= self._refit_every:
            self._step_count = 0
            self._refit()

    def _refit(self):
        """Fit A, B via ordinary least squares on the current buffer."""
        if len(self._obs_buf) < 10:
            return
        obs = np.array(self._obs_buf)       # [N, obs_dim]
        act = np.array(self._act_buf)       # [N, action_dim]
        nobs = np.array(self._next_obs_buf) # [N, obs_dim]

        # Design matrix: X @ W ≈ nobs
        X = np.concatenate([obs, act], axis=-1)  # [N, obs_dim + action_dim]
        W, _, _, _ = np.linalg.lstsq(X, nobs, rcond=None)
        # W : [obs_dim + action_dim, obs_dim]

        A = W[:self._obs_dim].astype(np.float32)   # [obs_dim, obs_dim]
        B = W[self._obs_dim:].astype(np.float32)   # [action_dim, obs_dim]

        self._A_var.assign(A)
        self._B_var.assign(B)

        mse = float(np.mean((X @ W - nobs) ** 2))
        print(f'[LinearSSM] Refit on {len(obs)} transitions. Train MSE: {mse:.6f}')

    # ------------------------------------------------------------------
    # RSSM-compatible interface
    # ------------------------------------------------------------------

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        return {'obs': tf.zeros([batch_size, self._obs_dim], dtype)}

    def observe(self, obs, action, state=None):
        """Process a full observed sequence.

        Args:
            obs   : [B, T, obs_dim]
            action: [B, T, action_dim]
        Returns:
            (post, prior) each with shape [B, T, obs_dim] under key 'obs'
        """
        if state is None:
            state = self.initial(tf.shape(action)[0])
        obs_t = tf.transpose(obs, [1, 0, 2])
        action_t = tf.transpose(action, [1, 0, 2])
        post, prior = tools.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (action_t, obs_t), (state, state))
        post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return post, prior

    def obs_step(self, prev_state, prev_action, obs):
        """Single posterior step: post = actual obs, prior = linear prediction."""
        prior = self.img_step(prev_state, prev_action)
        post = {'obs': obs}
        return post, prior

    def img_step(self, prev_state, prev_action):
        """Single imagination step using the fitted linear model."""
        dtype = prec.global_policy().compute_dtype
        obs = tf.cast(prev_state['obs'], dtype)
        action = tf.cast(prev_action, dtype)
        A = tf.cast(self._A_var, dtype)
        B = tf.cast(self._B_var, dtype)
        pred = obs @ A + action @ B
        return {'obs': pred}

    def get_feat(self, state):
        return state['obs']

    def get_dist(self, state):
        return tfd.Independent(
            tfd.Normal(state['obs'], tf.ones_like(state['obs'])), 1)

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        action_t = tf.transpose(action, [1, 0, 2])
        prior = tools.static_scan(self.img_step, action_t, state)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior


class MLPDynamics(tools.Module):
    """
    MLP dynamics model trained online with MSE loss.

    Architecture: concat(obs, action) → Dense(256, ELU) × 2 → Dense(obs_dim)
    Predicts the *next* observation.
    """

    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_size = hidden_size

    # ------------------------------------------------------------------
    # Forward pass (used both for loss and imagination)
    # ------------------------------------------------------------------

    def _predict(self, obs, action):
        """Predict next obs from current (obs, action) pair.

        Inputs can be any leading batch dimensions; obs and action are cast to
        the global compute dtype inside this method.
        """
        dtype = prec.global_policy().compute_dtype
        x = tf.concat([tf.cast(obs, dtype), tf.cast(action, dtype)], -1)
        x = self.get('h1', tfkl.Dense, self._hidden_size, tf.nn.elu)(x)
        x = self.get('h2', tfkl.Dense, self._hidden_size, tf.nn.elu)(x)
        x = self.get('out', tfkl.Dense, self._obs_dim, None)(x)
        return x

    def compute_loss(self, obs, actions):
        """MSE loss over all consecutive pairs in a [B, T, *] batch."""
        pred = self._predict(obs[:, :-1], actions[:, :-1])
        target = tf.cast(obs[:, 1:], prec.global_policy().compute_dtype)
        return tf.reduce_mean(tf.square(pred - target))

    # ------------------------------------------------------------------
    # RSSM-compatible interface
    # ------------------------------------------------------------------

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        return {'obs': tf.zeros([batch_size, self._obs_dim], dtype)}

    def observe(self, obs, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        obs_t = tf.transpose(obs, [1, 0, 2])
        action_t = tf.transpose(action, [1, 0, 2])
        post, prior = tools.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (action_t, obs_t), (state, state))
        post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return post, prior

    def obs_step(self, prev_state, prev_action, obs):
        """Post = actual obs; prior = MLP prediction from previous state."""
        prior = self.img_step(prev_state, prev_action)
        post = {'obs': obs}
        return post, prior

    def img_step(self, prev_state, prev_action):
        obs = prev_state['obs']
        pred = self._predict(obs, prev_action)
        return {'obs': pred}

    def get_feat(self, state):
        return state['obs']

    def get_dist(self, state):
        return tfd.Independent(
            tfd.Normal(state['obs'], tf.ones_like(state['obs'])), 1)

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        action_t = tf.transpose(action, [1, 0, 2])
        prior = tools.static_scan(self.img_step, action_t, state)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior
