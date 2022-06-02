from importlib_metadata import Distribution
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools
tfl = tf.keras.layers
LN2 = 0.69314718056

def flatten(x):
    return tf.reshape(x, [-1] + list(x.shape[2:]))

class RSSM(tools.Module):

  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2])
    action = tf.transpose(action, [1, 0, 2])
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),(action, embed), (state, state)
    )
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action):

    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior


class ConvEncoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs,image_source='image'):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs[image_source], (-1,) + tuple(obs[image_source].shape[-3:]))
    x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)
    x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)
    shape = tf.concat([tf.shape(obs[image_source])[:-3], [32 * self._depth]], 0)
    return tf.reshape(x, shape)


class ConvDecoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
    self._act = act
    self._depth = depth
    self._shape = shape

  def __call__(self, features):
    kwargs = dict(strides=2, activation=self._act)
    x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
    x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseDecoder(tools.Module):

  def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
    self._shape = shape
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act

  def __call__(self, features):
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if self._dist == 'normal':
      return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(logits = x), len(self._shape)), x
    raise NotImplementedError(self._dist)


class ActionDecoder(tools.Module):

  def __init__(
      self, size, layers, units, dist='tanh_normal', act=tf.nn.elu,
      min_std=1e-4, init_std=5, mean_scale=5):
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._mean_scale = mean_scale

  def __call__(self, features):
    raw_init_std = np.log(np.exp(self._init_std) - 1)
    x = features
    dist =None
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      mean, std = tf.split(x, 2, -1)
      mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
      std = tf.nn.softplus(std + raw_init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      dist = tools.OneHotDist(x)
    else:
      raise NotImplementedError(dist)
    return dist

class StatisticsNet(tf.keras.layers.Layer):
    """Statistics network model."""

    def __init__(self, mi_layers):
        super(StatisticsNet, self).__init__()
        self._mi_layers = mi_layers

    def layers_out(self, inputs):
        out = inputs
        for layer in self._mi_layers:
            out = layer(out)
        return out

    @tf.function
    def call(self, inputs):
        out = tf.concat(inputs, axis=-1)
        score = self.layers_out(out)
        return score

class MIEstimator(tools.Module):
    """Statistics network model for MI estimation."""
    def __init__(self,
                label = True,
                epochs = 1,
                batch_size = 64*50,
                prior_mi_constant = 1.0,
                max_mi_prior = 0.001,
                min_mi_prior_constant = 1e-3,
                negative_priors = True,
                mi_lagrangian_optimizer=tf.keras.optimizers.Adam(1e-3),
                mi_optimizer=tf.keras.optimizers.Adam(1e-3),
    ):
        self.label = label
        self.epochs = epochs
        self.batch_size = batch_size
        self._units = 64
        self._act = tf.nn.tanh
        self._shape = ()
        self._layers = 2
        self._unbiased_mi = True
        self._unbiased_mi_decay = .99
        self._clip_mi_predictions = True
        self._use_min_double_mi = True
        self.m_updates_per_d=1
        self.pr_mi = []
        self._mi_prior_constant = prior_mi_constant
        self._max_mi_prior = max_mi_prior

        d_mi_hidden_units = [256,256]
        mi_layers = [tfl.Dense(units, activation='tanh') for units in d_mi_hidden_units]
        mi_layers.append(tfl.Dense(1))
        self._mi_est = StatisticsNet(mi_layers)
        self._mi2_est = StatisticsNet(mi_layers)


        if self._mi_prior_constant > 0.0:
          # 获取 self._mi_est 以及 self._train_mi
          # 获取 self._mi2_est 以及 self._train_mi2
            if self._unbiased_mi:
                self._unbiased_mi_ma = tf.Variable(1.0, trainable=False)
            else:
                self._unbiased_mi_ma = None
            self._train_mi = self._make_mi_training_op(self._mi_est, mi_optimizer,
                                                       self._unbiased_mi_ma)
            if self._unbiased_mi:
                self._unbiased_mi_ma2 = tf.Variable(1.0, trainable=False)
            else:
                self._unbiased_mi_ma2 = None
            self._train_mi2 = self._make_mi_training_op(self._mi2_est, mi_optimizer,
                                                        self._unbiased_mi_ma2)
            self._double_mi = True

        if self._mi_prior_constant > 0.0:
            self._log_min_mi_prior_constant = tf.math.log(min_mi_prior_constant)
            self._prior_domains_data = True
            self._log_mi_prior_constant = tf.Variable(tf.math.log(prior_mi_constant))
            self.update_dual_mi_prior_constant = self._make_dual_mi_constant_update(
                log_mi_constant=self._log_mi_prior_constant,
                max_mi=self._max_mi_prior,
                optimizer=mi_lagrangian_optimizer
            )
        else:
            self._prior_domains_data = False
        self._negative_priors = negative_priors
    
    def train(self, l_pre_data, e_pre_data):
      input_shape = l_pre_data.shape
      l_pre_data = tf.reshape(l_pre_data,(-1,)+tuple(input_shape[2:]))
      e_pre_data = tf.reshape(e_pre_data,(-1,)+tuple(input_shape[2:]))
      n_batchs =  int(l_pre_data.shape[0]/self.batch_size)
      if self._mi_prior_constant > 0.0  or self._prior_domains_data:
        for epoch in range(self.epochs):
          for i in range(n_batchs):
            l_pre_batch = l_pre_data[i * self.batch_size:(i + 1) * self.batch_size]
            e_pre_batch = e_pre_data[i * self.batch_size:(i + 1) * self.batch_size]
            mi_prior_loss = self._train_mi(l_pre_batch, e_pre_batch)
            mi_prior_est = -1 * mi_prior_loss
            if self._double_mi:
              mi2_prior_loss = self._train_mi2(l_pre_batch, e_pre_batch)
              mi2_prior_est = -1 * mi2_prior_loss
              mi_prior_est = tf.maximum(mi_prior_est, mi2_prior_est)
              if self._clip_mi_predictions:
                  mi_prior_est = tf.clip_by_value(mi_prior_est, 0.0, 1.0)
              self.pr_mi.append(mi_prior_est)
            self.update_dual_mi_prior_constant(mi_prior_est)
            self._log_mi_prior_constant.assign(tf.maximum(self._log_mi_prior_constant,
                                                          self._log_min_mi_prior_constant))
            self._mi_prior_constant = tf.exp(self._log_mi_prior_constant)
      return self.pr_mi

    @staticmethod
    def _dv_kl(est, p_samples, q_samples):
        p_samples_estimate = tf.reduce_mean(est(p_samples))
        q_samples_estimate = tf.math.log(tf.reduce_mean(tf.exp(est(q_samples))))
        return (p_samples_estimate - q_samples_estimate) / LN2

    @staticmethod
    def _get_mi_batches_label(l_pre_batch, e_pre_batch):
        l_pre_batch_shape = l_pre_batch.get_shape()
        e_pre_batch_shape = e_pre_batch.get_shape()
        l_pre_batch_n = l_pre_batch_shape[0]
        e_pre_batch_n = e_pre_batch_shape[0]
        l_pre_batch = tf.reshape(l_pre_batch, [e_pre_batch_n, -1])
        e_pre_batch = tf.reshape(e_pre_batch, [l_pre_batch_n, -1])
        input_correct_batch = tf.concat([l_pre_batch, e_pre_batch], axis=0)
        domain_labels = tf.concat([tf.zeros([l_pre_batch_n, 1]),
                                   tf.ones([e_pre_batch_n, 1])], axis=0)
        shuffled_domain_labels = tf.random.shuffle(domain_labels)
        positive_ordering = tf.concat([input_correct_batch, domain_labels],
                                      axis=1)
        negative_ordering = tf.concat([input_correct_batch, shuffled_domain_labels],
                                      axis=1)
        return positive_ordering, negative_ordering

    @staticmethod
    def _get_mi_batches(l_pre_batch, e_pre_batch):
        positive_ordering = tf.concat([l_pre_batch,e_pre_batch],axis=-1)
        #random_e_pre_batch = tf.random.shuffle(e_pre_batch)
        random_e_pre_batch = tf.gather(e_pre_batch, tf.random.shuffle(tf.range(tf.shape(input=e_pre_batch)[0])))
        negative_ordering = tf.concat([l_pre_batch,random_e_pre_batch],axis=-1)
        return positive_ordering, negative_ordering

    def _mi_loss(self, mi_est, l_pre_batch, e_pre_batch):
        if self.label:
          positive_ordering, negative_ordering = self._get_mi_batches_label(l_pre_batch, e_pre_batch)
        else:
          positive_ordering, negative_ordering = self._get_mi_batches(l_pre_batch, e_pre_batch)

        return -1 * self._dv_kl(mi_est, positive_ordering, negative_ordering)

    def _make_mi_training_op(self, mi_est, optimizer, mi_ma=None):
        if mi_ma is None:
            def train(l_pre_batch, e_pre_batch):
                with tf.GradientTape() as tape:
                    mi_loss = self._mi_loss(mi_est, l_pre_batch, e_pre_batch)
                    gradients = tape.gradient(mi_loss, mi_est.trainable_weights)
                optimizer.apply_gradients(zip(gradients, mi_est.trainable_weights))
                return mi_loss
        else:
            def loss_fn(mi_est, l_pre_batch, e_pre_batch):
                if self.label:
                  p_samples, q_samples = self._get_mi_batches_label(l_pre_batch, e_pre_batch)
                else:
                  p_samples, q_samples = self._get_mi_batches(l_pre_batch, e_pre_batch)
                p_samples_estimate = tf.reduce_mean(mi_est(p_samples))
                batch_q_exp_samples_estimate = tf.reduce_mean(tf.exp(mi_est(q_samples)))
                mi_ma.assign(tf.stop_gradient(self._unbiased_mi_decay * mi_ma +
                                              (1 - self._unbiased_mi_decay) *
                                              batch_q_exp_samples_estimate))
                unbiased_loss = -(p_samples_estimate - batch_q_exp_samples_estimate / mi_ma) / LN2
                mi_loss = -(p_samples_estimate - tf.math.log(batch_q_exp_samples_estimate)) / LN2
                return unbiased_loss, mi_loss

            def train(l_pre_batch, e_pre_batch):
                with tf.GradientTape() as tape:
                    unbiased_loss, mi_loss = loss_fn(mi_est, l_pre_batch, e_pre_batch)
                    gradients = tape.gradient(unbiased_loss, mi_est.trainable_weights)
                optimizer.apply_gradients(zip(gradients, mi_est.trainable_weights))
                return mi_loss
        return tf.function(train)

    def get_mi_loss(self,l_prior_latent_batch,e_prior_latent_batch):
      input_shape = l_prior_latent_batch.shape
      l_prior_latent_batch = tf.reshape(l_prior_latent_batch,(-1,)+tuple(input_shape[2:]))
      e_prior_latent_batch = tf.reshape(e_prior_latent_batch,(-1,)+tuple(input_shape[2:]))
      prior_mi = self.get_prior_mi(l_prior_latent_batch, e_prior_latent_batch)
      mi_loss = self.get_prior_mi_constant() * prior_mi
      return mi_loss
    
    def get_prior_mi_constant(self):
      return tf.exp(self._log_mi_prior_constant)
    
    def get_prior_mi(self,l_prior_pre_batch, e_prior_pre_batch):
                    return tf.math.maximum(
                        -1 * tf.math.minimum(self._mi_loss(self._mi_est, l_prior_pre_batch, e_prior_pre_batch),
                                             self._mi_loss(self._mi2_est, l_prior_pre_batch, e_prior_pre_batch)),
                        0.0)

    
    @staticmethod
    def _make_dual_mi_constant_update(log_mi_constant, max_mi, optimizer):
        def update_dual_mi_constant(mi_estimate):
            mi_diff = max_mi - mi_estimate
            with tf.GradientTape() as tape:
                mi_dual_loss = log_mi_constant * tf.stop_gradient(mi_diff)
                gradients = tape.gradient(mi_dual_loss, [log_mi_constant])
            optimizer.apply_gradients(zip(gradients, [log_mi_constant]))

        return update_dual_mi_constant

