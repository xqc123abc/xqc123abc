import numpy as np
import tensorflow as tf
from tensorflow import array_ops, init_ops, math_ops
from tensorflow.contrib import layers, rnn
import gc


def expand(x, axis, N, dims=2):
    if dims != 2:
        return tf.tile(tf.expand_dims(x, axis), [N, 1, 1])
    return tf.tile(tf.expand_dims(x, axis), [N, 1])
    # return tf.concat([tf.expand_dims(x, dim) for _ in tf.range(N)], axis=dim)


def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def learned_init(units):
    return tf.squeeze(layers.fully_connected(
        tf.ones([1, 1]), units, activation_fn=None, biases_initializer=None))


class MIMNCell(rnn.RNNCell):
    def __init__(self, controller_units, memory_vector_dim, batch_size=128, memory_size=4,
                 read_head_num=1, write_head_num=1, reuse=False, output_dim=16, clip_value=20, sharp_value=2.):
        super(MIMNCell, self).__init__()
        self.controller_units = controller_units
        self.memory_vector_dim = memory_vector_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.reuse = reuse
        self.clip_value = clip_value
        self.sharp_value = sharp_value

        def single_cell(num_units):
            return rnn.GRUCell(num_units)

        self.controller = single_cell(self.controller_units)
        self.step = 0
        self.output_dim = output_dim

        # 初始化参数
        self.o2p_initializer = create_linear_initializer(self.controller_units)
        self.o2o_initializer = create_linear_initializer(
            self.controller_units + self.memory_vector_dim * self.read_head_num)

    @property
    def state_size(self):
        return {
            "controller_state": self.controller.state_size,
            "read_vector_list": [self.memory_vector_dim for _ in range(self.read_head_num)],
            "w_list": self.read_head_num + self.write_head_num,
            "M": (self.memory_size, self.memory_vector_dim),
            "key_M": (self.memory_size, self.memory_vector_dim),
            "sum_aggre": (self.memory_size, self.memory_vector_dim)
        }

    @property
    def output_size(self):
        return self.output_dim

    def __call__(self, x, prev_state, scope=None):
        prev_read_vector_list = prev_state["read_vector_list"]

        controller_input = tf.concat([x] + prev_read_vector_list, axis=1)
        with tf.variable_scope('controller', reuse=self.reuse or tf.AUTO_REUSE):
            controller_output, controller_state = self.controller(controller_input, prev_state["controller_state"])

        num_parameters_per_head = self.memory_vector_dim + 1  # TODO: 为什么 +1？ 是 sharp_value 的原因吗？
        num_heads = self.read_head_num + self.write_head_num
        total_parameter_num = num_parameters_per_head * num_heads + self.memory_vector_dim * 2 * self.write_head_num

        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse or tf.AUTO_REUSE):
            parameters = layers.fully_connected(
                controller_output, total_parameter_num, activation_fn=None,
                weights_initializer=self.o2p_initializer, biases_initializer=tf.zeros_initializer())
            parameters = tf.clip_by_norm(parameters, self.clip_value)

        head_parameter_list = tf.split(parameters[:, :num_parameters_per_head * num_heads], num_heads, axis=1)
        erase_add_list = tf.split(parameters[:, num_parameters_per_head * num_heads:], 2 * self.write_head_num, axis=1)

        prev_M = prev_state["M"]  # shape: [batch_size, memory_size, memory_vector_dim]
        key_M = prev_state["key_M"]  # shape: [batch_size, memory_size, memory_vector_dim]
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim])
            beta = (tf.nn.softplus(head_parameter[:, self.memory_vector_dim]) + 1) * self.sharp_value
            with tf.variable_scope(f'addressing_head_{i}', reuse=tf.AUTO_REUSE):
                w = self.addressing(k, beta, key_M, prev_M)  # [batch_size, memory_size]
            w_list.append(w)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            # [batch_size, memory_vector_dim]
            read_vector = tf.reduce_sum(tf.expand_dims(read_w_list[i], axis=2) * prev_M, axis=1)
            read_vector = tf.reshape(read_vector, [-1, self.memory_vector_dim])
            read_vector_list.append(read_vector)

        write_w_list = w_list[self.read_head_num:]

        M = prev_M
        sum_aggre = prev_state["sum_aggre"]

        for i in range(self.write_head_num):
            w = tf.expand_dims(write_w_list[i], axis=2)  # [batch_size, memory_size, 1]
            erase_vector = tf.expand_dims(tf.sigmoid(erase_add_list[i * 2]), axis=1)  # [batch_size, 1, memory_vector_dim]
            add_vector = tf.expand_dims(tf.tanh(erase_add_list[i * 2 + 1]), axis=1)  # [batch_size, 1, memory_vector_dim]

            # [batch_size, memory_size, memory_vector_dim]
            # M_t = (1 - E_t) * M_t + A_t
            ones = tf.ones([self.batch_size, self.memory_size, self.memory_vector_dim])
            M = M * (ones - tf.matmul(w, erase_vector)) + tf.matmul(w, add_vector)
            sum_aggre += tf.matmul(tf.stop_gradient(w), add_vector)  # [batch_size, memory_size, memory_vector_dim]

        with tf.variable_scope("o2o", reuse=(self.step > 0) or self.reuse or tf.AUTO_REUSE):
            read_output = layers.fully_connected(
                tf.concat([controller_output] + read_vector_list, axis=1), self.output_dim, activation_fn=None,
                weights_initializer=self.o2o_initializer, biases_initializer=tf.zeros_initializer())
            read_output = tf.clip_by_norm(read_output, self.clip_value)

        self.step += 1
        return read_output, {
            "controller_state": controller_state,
            "read_vector_list": read_vector_list,
            "w_list": w_list,
            "M": M,
            "key_M": key_M,
            "sum_aggre": sum_aggre
        }

    def addressing(self, k, beta, key_M, prev_M):
        # 余弦相似度
        def cosine_similarity(key, M):
            key = tf.expand_dims(key, axis=2)  # [batch_size, memory_vector_dim, 1]
            inner_product = tf.matmul(M, key)  # [batch_size, memory_size, 1]
            k_norm = tf.sqrt(tf.reduce_sum(tf.square(key), axis=1, keepdims=True))  # [batch_size, 1, 1]
            M_norm = tf.sqrt(tf.reduce_sum(tf.square(M), axis=2, keepdims=True))  # [batch_size, memory_size, 1]
            norm_product = M_norm * k_norm  # [batch_size, memory_size, 1]
            K = tf.squeeze(inner_product / (norm_product + 1e-8), axis=2)  # [batch_size, memory_size]
            return K

        K = 0.5 * (cosine_similarity(k, key_M) + cosine_similarity(k, prev_M))
        K_amplified = tf.exp(tf.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / tf.reduce_sum(K_amplified, axis=1, keepdims=True)  # [batch_size, memory_size]

        return w_c

    def zero_state(self, batch_size):
        with tf.variable_scope('init', reuse=self.reuse or tf.AUTO_REUSE):
            read_vector_list = [expand(tf.tanh(learned_init(self.memory_vector_dim)), 0, batch_size)
                                for _ in range(self.read_head_num)]

            w_list = [expand(tf.nn.softmax(learned_init(self.memory_size)), 0, batch_size)
                      for _ in range(self.read_head_num + self.write_head_num)]

            controller_init_state = self.controller.zero_state(batch_size, tf.float32)

            # 初始化 memory M 和 key_M
            M = expand(tf.tanh(tf.get_variable(
                'init_M', [self.memory_size, self.memory_vector_dim],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-5), trainable=False)), 0, batch_size, 3)

            key_M = expand(tf.tanh(tf.get_variable(
                    'key_M', [self.memory_size, self.memory_vector_dim],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))), 0, batch_size, 3)

            sum_aggre = tf.zeros([batch_size, self.memory_size, self.memory_vector_dim], dtype=tf.float32)

            state = {
                "controller_state": controller_init_state,
                "read_vector_list": read_vector_list,
                "w_list": w_list,
                "M": M,
                "key_M": key_M,
                "sum_aggre": sum_aggre
            }
            return state


class VecAttGRUCell(rnn.RNNCell):
    """带有向量注意力的 GRU 单元（参考：http://arxiv.org/abs/1406.1078）。
    Args:
      num_units: int, GRU 单元的数量。
      activation: 激活函数。默认是 `tanh`。
      reuse: (可选) 布尔值，是否复用已有变量。
      kernel_initializer: (可选) 权重和投影矩阵的初始化器。
      bias_initializer: (可选) 偏置项的初始化器。
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__()
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """带有注意力分数的 GRU 单元."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with tf.variable_scope("gates", reuse=tf.AUTO_REUSE):
                self._gate_linear = layers.fully_connected(
                    tf.concat([inputs, state], axis=1),
                    2 * self._num_units,
                    activation_fn=tf.sigmoid,
                    weights_initializer=self._kernel_initializer,
                    biases_initializer=bias_ones)

        value = math_ops.sigmoid(self._gate_linear(tf.concat([inputs, state], axis=1)))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with tf.variable_scope("candidate", reuse=tf.AUTO_REUSE):
                self._candidate_linear = layers.fully_connected(
                    tf.concat([inputs, r_state], axis=1),
                    self._num_units,
                    activation_fn=self._activation,
                    weights_initializer=self._kernel_initializer,
                    biases_initializer=self._bias_initializer)
        c = self._activation(self._candidate_linear(tf.concat([inputs, r_state], axis=1)))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h
