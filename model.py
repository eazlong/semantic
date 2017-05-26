# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
# from tensorflow.models.rnn import rnn
import reader
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBModel(object):
  """The PTB model."""

  def __init__(self, g, is_training, config, category, class_number=4, inited=False):
    print(tf.get_default_graph())
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    keep_prob = config.keep_prob
    layer_num = config.num_layers
    # self.is_training = is_training

    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope(category, reuse=inited, initializer=initializer), g.name_scope(category):
      self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
      self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

      with tf.device("/cpu:0"), tf.variable_scope(category):
        embedding = tf.get_variable(
            "embedding", [vocab_size, size], dtype=data_type())

      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

      '''bldirectional_rnn'''
      # # ** 1.LSTM 层
      # lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
      # lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
      # # ** 2.dropout
      # lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
      #     cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
      # lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
      #     cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
      # # ** 3.多层 LSTM
      # cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * layer_num, state_is_tuple=True)
      # cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * layer_num, state_is_tuple=True)
      # # ** 4.初始状态
      # initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
      # initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

      # inputs_list = [tf.squeeze(s, [1]) for s in tf.split(1, num_steps, inputs)]

      # try:
      #   outputs, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs_list,
      #                                           initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype=tf.float32)
      # except Exception:  # Old TensorFlow version only returns outputs not states
      #   outputs = tf.nn.bidirectional_rnn(
      #       cell_fw, cell_bw, inputs_list, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype=tf.float32)
      # output = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size * 2])

      # softmax_w = tf.get_variable(
      #     "softmax_w", [size * 2, class_number], dtype=data_type())
      # softmax_b = tf.get_variable("softmax_b", [class_number], dtype=data_type())
      # logits = tf.matmul(output, softmax_w) + softmax_b

      # correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32),
      #                               tf.reshape(self._targets, [-1]))
      # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      #     labels=tf.reshape(self._targets, [-1]), logits=logits)

      '''lstm'''
      # Slightly better results can be obtained with forget gate biases
      # initialized to 1 but the hyperparameters of the model would need to be
      # different than reported in the paper.
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
      if is_training and config.keep_prob < 1:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=config.keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

      self._initial_state = cell.zero_state(batch_size, data_type())

      if is_training and config.keep_prob < 1:
        inputs = tf.nn.dropout(inputs, config.keep_prob)

      # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
      # This builds an unrolled LSTM for tutorial purposes only.
      # In general, use the rnn() or state_saving_rnn() from rnn.py.
      #
      # The alternative version of the code below is:
      #

      # inputs = [tf.squeeze(input_, [1])
      #           for input_ in tf.split(1, num_steps, inputs)]
      # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
      outputs = []
      state = self._initial_state
      with tf.variable_scope(category):
        for time_step in range(num_steps):
          if time_step > 0:
            tf.get_variable_scope().reuse_variables()
          (cell_output, state) = cell(inputs[:, time_step, :], state)
          outputs.append(cell_output)

      output = tf.reshape(tf.concat(1, outputs), [-1, size])

      with tf.variable_scope(category):
        softmax_w = tf.get_variable(
            "softmax_w", [size, class_number], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [class_number], dtype=data_type())
      logits = tf.matmul(output, softmax_w) + softmax_b

      correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32),
                                    tf.reshape(self._targets, [-1]))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.reshape(self._targets, [-1]), logits=logits)

      # loss = tf.nn.seq2seq.sequence_loss_by_example(
      #     logits=[logits],
      #     targets=[tf.reshape(self._targets, [-1])],
      #     weights=[tf.ones([batch_size * num_steps], dtype=data_type())])

      cost = tf.reduce_sum(loss) / batch_size  # loss [time_step]
      self._cost = cost
      self._final_state = state
      self._logits = logits

      # loss = tf.nn.seq2seq.sequence_loss_by_example(
      #     [logits],
      #     [tf.reshape(self._targets, [-1])],
      #     [tf.ones([batch_size * num_steps], dtype=data_type())])
      # self._cost = cost = tf.reduce_sum(loss) / batch_size
      # self._logits = logits
      if not is_training:
        return

      with tf.variable_scope(category):
        self._lr = tf.Variable(0.0, trainable=False, name="lr")
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        config.max_grad_norm)
      #optimizer = tf.train.GradientDescentOptimizer(self._lr)
      optimizer = tf.train.AdamOptimizer(self._lr)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars))

      self._new_lr = tf.placeholder(
          tf.float32, shape=[], name="new_learning_rate")
      self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def final_state(self):
    return self._final_state

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def cost(self):
    return self._cost

  @property
  def logits(self):
    return self._logits

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  # @property
  # def is_training(self):
  #   return self.is_training


class SmallConfig(object):
  """Small config."""
  init_scale = 0.04
  learning_rate = 0.6
  max_grad_norm = 10
  num_layers = 2
  num_steps = 12
  hidden_size = 128
  max_epoch = 5
  max_max_epoch = 55
  keep_prob = 0.8
  lr_decay = 1 / 1.15
  batch_size = 1
  vocab_size = 2000


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  predict_id = []
  for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                model.num_steps)):
    fetches = [model.cost, model.final_state, model.logits, eval_op]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    cost, state, logits, _ = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps
    predict_id.append(int(np.argmax(logits)))

    # if verbose:  # and step % (epoch_size // 10) == 10:
    #   print("%.3f perplexity: %.3f speed: %.0f wps" %
    #         (step * 1.0 / epoch_size, np.exp(costs / iters),
    #          iters * model.batch_size / (time.time() - start_time)))

    # Save Model to CheckPoint when is_training is True
    # if model.is_training:
    #   if step % (epoch_size // 10) == 10:
    #     checkpoint_path = os.path.join(FLAGS.pos_train_dir, "pos.ckpt")
    #     model.saver.save(session, checkpoint_path)
    #     print("Model Saved... at time step " + str(step))

  return np.exp(costs / iters), predict_id


# def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
#   """Runs the model on the given data."""
#   epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps

#   start_time = time.time()
#   costs = 0.0
#   iters = 0

#   predict_id = []
#   for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
#                                                 model.num_steps)):
#     fetches = [model.cost, model.logits, eval_op]  # eval_op define the m.train_op or m.eval_op
#     feed_dict = {}
#     feed_dict[model.input_data] = x
#     feed_dict[model.targets] = y
#     cost, logits, _ = session.run(fetches, feed_dict)
#     costs += cost
#     iters += model.num_steps
#     print(logits)
#     predict_id.append(int(np.argmax(logits)))

#   return np.exp(costs / iters), predict_id


class Predictor:

  def __init__(self, category, label_size, ckpt_path, step):
    print("Step", step)
    self._config = get_config()
    self._config.num_steps = step
    self._category = category
    self.inited = False
    self._checkpoint_dir = ckpt_path
    self.label_size = label_size
    self.g = tf.Graph()
    self.__define_models()

  def __define_models(self):
    with self.g.as_default():
      self.train_model = PTBModel(self.g, is_training=True, config=self._config,
                                  category=self._category, class_number=self.label_size, inited=False)

      eval_config = get_config()
      eval_config.batch_size = 1
      eval_config.num_steps = 1
      # 初始化类
      self.predict_model = PTBModel(self.g, is_training=False, config=eval_config, category=self._category,
                                    class_number=self.label_size, inited=True)

  def __init_model(self):
    if not self.inited:
      self.predict_session = tf.InteractiveSession()
      tf.initialize_all_variables().run()
      ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        tf.train.Saver().restore(self.predict_session, ckpt.model_checkpoint_path)
      else:
        raise ValueError("check point file %s not found. please train model first" %
                         self._checkpoint_dir)
    self.inited = True

  def train(self, train_data, vocab_list, retrain=False):
    with self.g.as_default():
      with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(self._config.max_max_epoch):
          lr_decay = self._config.lr_decay ** max(i - self._config.max_epoch, 0.0)
          self.train_model.assign_lr(sess, self._config.learning_rate * lr_decay)

          rarray = list(np.random.random(size=len(train_data)) * (len(train_data) - 1))
          rarray = [int(x) for x in rarray]
          word_data = []
          target_data = []
          for j in rarray:
            x = train_data[j]
            word_data.extend(x[0])
            target_data.extend(x[1])

          train_perplexity, _ = run_epoch(sess, self.train_model, word_data,
                                          target_data, self.train_model.train_op, verbose=True)
          print("Epoch: %d Learning rate: %.3f, perplexity: %.3f" %
                (i + 1, sess.run(self.train_model.lr), train_perplexity))

        # 训练完成，保存模型
        tf.train.Saver().save(sess, os.path.join(self._checkpoint_dir, self._category + ".ckpt"))
      sess.close()

  def predict(self, sentense):
    with self.g.as_default():
      self.__init_model()
      sess = self.predict_session
      labels = [0] * len(sentense)
      p, predict_ids = run_epoch(sess, self.predict_model, sentense, labels, tf.no_op())
      print(p, predict_ids)
      return predict_ids

if __name__ == "__main__":
  tf.app.run()
