# Mert Hacıahmetoğlu
# 03.08.2019

"""moduls script"""

#---------------------------------------------------------------------
from __future__ import print_function, division
import tensorflow as tf
#------------------------------------------------------

#embedding fonksiyonu:
def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):

  with tf.variable_scope(scope, reuse=reuse):
    lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, num_units],
                                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    if zero_pad:
      lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)

    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
  return outputs

def normalize(inputs,scope="normalize",reuse=None):
  #katman normalizasyonu
  outputs = tf.contrib.layers.layer_norm(inputs,begin_norm_axis=-1,scope=scope,reuse=reuse)
  return outputs

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
  # Highway networks, see https://arxiv.org/abs/1505.00387

  if not num_units:
    num_units = inputs.get_shape()[-1]

  with tf.variable_scope(scope, reuse=reuse):
    H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
    T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                        bias_initializer=tf.constant_initializer(-1.0), name="dense2")
    outputs = H * T + inputs * (1. - T)
  return outputs

def conv1d(inputs, filters=None, size=1, rate=1, padding="SAME", dropout_rate=0, use_bias=True,
           activation_fn=None, training=True, scope="conv1d", reuse=None):
  with tf.variable_scope(scope):
    if padding.lower() == "causal":
      pad_len = (size - 1) * rate  # padding size
      inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
      padding = "valid"

    if filters is None:
      filters = inputs.get_shape().as_list()[-1]

    params = {"inputs": inputs, "filters": filters, "kernel_size": size,
              "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
              "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}

    tensor = tf.layers.conv1d(**params)
    tensor = normalize(tensor)
    if activation_fn is not None:
      tensor = activation_fn(tensor)

    tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)
  return tensor

def hc(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       dropout_rate=0,
       use_bias=True,
       activation_fn=None,
       training=True,
       scope="hc",
       reuse=None):
  _inputs = inputs
  with tf.variable_scope(scope):
    if padding.lower() == "causal":
      pad_len = (size - 1) * rate
      inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
      padding = "valid"
    if filters is None:
      filters = inputs.get_shape().as_list()[-1]

    params = {"inputs": inputs, "filters": 2*filters, "kernel_size": size,
              "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
              "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(), "reuse": reuse}
    tensor = tf.layers.conv1d(**params)
    H1, H2 = tf.split(tensor, 2, axis=-1)
    H1 = normalize(H1, scope="H1")
    H2 = normalize(H2, scope="H2")
    H1 = tf.nn.sigmoid(H1, "gate")
    H2 = activation_fn(H2, "info") if activation_fn is not None else H2

    tensor = H1*H2 + (1.-H1)*_inputs
    tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)
  return tensor

def conv1d_transpose(inputs,
                     filters=None,
                     size=3,
                     stride=2,
                     padding='same',
                     dropout_rate=0,
                     use_bias=True,
                     activation=None,
                     training=True,
                     scope="conv1d_transpose",
                     reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    if filters is None:
      filters = inputs.get_shape().as_list()[-1]
    inputs = tf.expand_dims(inputs, 1)
    tensor = tf.layers.conv2d_transpose(inputs, filters=filters,  kernel_size=(1, size),
                                        strides=(1, stride), padding=padding, activation=None,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        use_bias=use_bias)
    tensor = tf.squeeze(tensor, 1)
    tensor = normalize(tensor)
    if activation is not None:
      tensor = activation(tensor)
    tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)
  return tensor