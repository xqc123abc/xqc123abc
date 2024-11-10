import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell
from utils import MIMNCell, VecAttGRUCell
from rnn import dynamic_rnn
from sklearn.cluster import KMeans
import numpy as np
from dataloader import DataLoader  

class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        # 重置计算图
        tf.reset_default_graph()

        # 输入占位符
        with tf.name_scope('inputs'):
            self.user_seq_ph = tf.placeholder(tf.int32, [None, max_time_len, item_fnum], name='user_seq_ph')
            self.user_seq_length_ph = tf.placeholder(tf.int32, [None,], name='user_seq_length_ph')
            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

            # 学习率
            self.lr = tf.placeholder(tf.float32, [])
            # 正则化系数
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # dropout 保留率
            self.keep_prob = tf.placeholder(tf.float32, [])

        # 词嵌入
        with tf.name_scope('embedding'):
            if emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=emb_initializer)
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer())
                self.emb_mtx_mask = tf.constant(value=1., shape=[feature_size - 1, eb_dim])
                self.emb_mtx_mask = tf.concat([tf.constant(value=0., shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
                self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

            self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_seq_ph)
            self.user_seq = tf.reshape(self.user_seq, [-1, max_time_len, item_fnum * eb_dim])
            self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph)
            self.target_item = tf.reshape(self.target_item, [-1, item_fnum * eb_dim])
            self.target_user = tf.nn.embedding_lookup(self.emb_mtx, self.target_user_ph)
            self.target_user = tf.reshape(self.target_user, [-1, user_fnum * eb_dim])

    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
        score = tf.nn.softmax(fc3)
        # 输出预测值
        self.y_pred = tf.reshape(score[:,0], [-1,])

    def build_logloss(self):
        # 损失函数
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # 优化器和训练步骤
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_mseloss(self):
        self.loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        # 正则化项
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # 优化器和训练步骤
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.user_seq_ph: batch_data[0],
            self.user_seq_length_ph: batch_data[1],
            self.target_user_ph: batch_data[2],
            self.target_item_ph: batch_data[3],
            self.label_ph: batch_data[4],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: 0.8
        })
        return loss

    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.user_seq_ph: batch_data[0],
            self.user_seq_length_ph: batch_data[1],
            self.target_user_ph: batch_data[2],
            self.target_item_ph: batch_data[3],
            self.label_ph: batch_data[4],
            self.reg_lambda: reg_lambda,
            self.keep_prob: 1.0
        })
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('模型已从 {} 恢复'.format(path))

def train_din_and_get_item_embeddings(dataset_size, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, learning_rate, reg_lambda, batch_size, user_seq_file, target_train_file, user_feat_dict_file, item_feat_dict_file):
    # 初始化 DIN 模型
    din_model = DIN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)

    # 创建会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        train_losses_step = []
        early_stop = False
        eval_iter_num = (dataset_size // 5) // batch_size  # 请确保在函数调用中传入 dataset_size

        # 开始训练过程
        for epoch in range(1):  # 可以调整 epoch 数量
            if early_stop:
                break

            data_loader = DataLoader(batch_size, user_seq_file, target_train_file, user_feat_dict_file, item_feat_dict_file, max_time_len)

            for batch_data in data_loader:
                if early_stop:
                    break

                loss = din_model.train(sess, batch_data, learning_rate, reg_lambda)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses_step = []
                    # 假设我们有一个简单的早停机制
                    if len(train_losses_step) > 2 and epoch > 0:
                        if (train_losses_step[-1] > train_losses_step[-2] and train_losses_step[-2] > train_losses_step[-3]):
                            early_stop = True
                        if (train_losses_step[-2] - train_losses_step[-1]) <= 0.001 and (train_losses_step[-3] - train_losses_step[-2]) <= 0.001:
                            early_stop = True

        # 推理阶段，提取商品嵌入
        item_embeddings = []
        for item_id in range(feature_size):
            # 构造输入数据
            item_input = np.zeros((1, item_fnum))
            item_input[0, 0] = item_id  # 假设 item_id 在 item_fnum 的第一个位置

            # 获取嵌入
            item_emb = sess.run(din_model.target_item, feed_dict={
                din_model.target_item_ph: item_input,
                din_model.keep_prob: 1.0
            })
            item_embeddings.append(item_emb.flatten())

    return np.array(item_embeddings)

class SumPooling(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(SumPooling, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)

        # 使用 sum pooling 来建模用户行为，填充为零（嵌入 ID 也是零）
        user_behavior_rep = tf.reduce_sum(self.user_seq, axis=1)

        inp = tf.concat([user_behavior_rep, self.target_item, self.target_user], axis=1)

        # 全连接层
        self.build_fc_net(inp)
        self.build_logloss()

class GRU4Rec(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(GRU4Rec, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)

        # GRU
        with tf.name_scope('rnn'):
            user_seq_ht, user_seq_final_state = tf.nn.dynamic_rnn(
                GRUCell(hidden_size),
                inputs=self.user_seq,
                sequence_length=self.user_seq_length_ph,
                dtype=tf.float32,
                scope='gru1'
            )
            # 如果需要，可以在这里添加更多的 RNN 层

        inp = tf.concat([user_seq_final_state, self.target_item, self.target_user], axis=1)

        # 全连接层
        self.build_fc_net(inp)
        self.build_logloss()

class Caser(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(Caser, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)

        with tf.name_scope('user_seq_cnn'):
            # 水平卷积核
            filters_user = 4
            h_kernel_size_user = [5, eb_dim * item_fnum]
            v_kernel_size_user = [self.user_seq.get_shape().as_list()[1], 1]

            self.user_seq = tf.expand_dims(self.user_seq, 3)
            conv1 = tf.layers.conv2d(self.user_seq, filters_user, h_kernel_size_user)
            max1 = tf.layers.max_pooling2d(conv1, [conv1.get_shape().as_list()[1], 1], 1)
            user_hori_out = tf.reshape(max1, [-1, filters_user])  # [B, F]

            # 垂直卷积核
            conv2 = tf.layers.conv2d(self.user_seq, filters_user, v_kernel_size_user)
            conv2 = tf.reshape(conv2, [-1, eb_dim * item_fnum, filters_user])
            user_vert_out = tf.reshape(tf.layers.dense(conv2, 1), [-1, eb_dim * item_fnum])

            inp = tf.concat([user_hori_out, user_vert_out, self.target_item, self.target_user], axis=1)

        # 全连接层
        self.build_fc_net(inp)
        self.build_logloss()

class DIN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(DIN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        mask = tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32)
        _, user_behavior_rep = self.attention(self.user_seq, self.user_seq, self.target_item, mask)

        inp = tf.concat([user_behavior_rep, self.target_user, self.target_item], axis=1)

        # 全连接层
        self.build_fc_net(inp)
        self.build_logloss()

    def attention(self, key, value, query, mask):
        # key: [B, T, Dk], query: [B, Dq], mask: [B, T]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        kq_inter = queries * key
        atten = tf.reduce_sum(kq_inter, axis=2)

        mask = tf.equal(mask, tf.ones_like(mask))  # [B, T]
        paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
        atten = tf.nn.softmax(tf.where(mask, atten, paddings))  # [B, T]
        atten = tf.expand_dims(atten, 2)

        res = tf.reduce_sum(atten * value, axis=1)
        return atten, res

class HPMN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(HPMN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        self.layer_num = 3
        self.split_by = 2
        self.memory = []
        with tf.name_scope('rnn'):
            inp = self.user_seq
            length = max_time_len
            for i in range(self.layer_num):
                user_seq_ht, user_seq_final_state = tf.nn.dynamic_rnn(
                    GRUCell(hidden_size),
                    inputs=inp,
                    dtype=tf.float32,
                    scope='GRU{}'.format(i)
                )

                user_seq_final_state = tf.expand_dims(user_seq_final_state, 1)
                self.memory.append(user_seq_final_state)

                length = int(length / self.split_by)
                user_seq_ht = tf.reshape(user_seq_ht, [-1, length, self.split_by, hidden_size])
                inp = tf.reshape(tf.gather(user_seq_ht, [self.split_by - 1], axis=2), [-1, length, hidden_size])

        self.memory = tf.concat(self.memory, axis=1)
        _, output = self.attention(self.memory, self.memory, self.target_item)
        self.repre = tf.concat([self.target_user, self.target_item, output], axis=1)
        self.build_fc_net(self.repre)
        self.build_loss()

    def attention(self, key, value, query):
        # key: [B, T, Dk], query: [B, Dq]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        kq_inter = queries * key
        atten = tf.reduce_sum(kq_inter, axis=2)

        atten = tf.nn.softmax(atten)  # [B, T]
        atten = tf.expand_dims(atten, 2)

        res = tf.reduce_sum(atten * value, axis=1)
        return atten, res

    def get_covreg(self, memory, k):
        mean = tf.reduce_mean(memory, axis=2, keepdims=True)
        C = memory - mean
        C = tf.matmul(C, tf.transpose(C, [0, 2, 1])) / tf.cast(tf.shape(memory)[2], tf.float32)
        C_diag = tf.linalg.diag_part(C)
        C_diag = tf.linalg.diag(C_diag)
        C = C - C_diag
        norm = tf.norm(C, ord='fro', axis=[1, 2])
        return tf.reduce_sum(norm)

    def build_loss(self):
        # 损失函数
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        # L2 正则化
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # 协方差正则化
        self.loss += self.reg_lambda * self.get_covreg(self.memory, self.layer_num)

        # 优化器和训练步骤
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

class MIMN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(MIMN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)

        with tf.name_scope('inputs'):
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size_ph')
        batch_size = self.batch_size

        cell = MIMNCell(hidden_size, item_fnum * eb_dim, batch_size)

        state = cell.zero_state(batch_size)

        for t in range(max_time_len):
            _, state = cell(self.user_seq[:, t, :], state)

        # [batch_size, memory_size, fnum * eb_dim] -> [batch_size, fnum * eb_dim]
        mean_memory = tf.reduce_mean(state['sum_aggre'], axis=1)

        read_out, _ = cell(self.target_item, state)

        self.item_his_eb_sum = tf.reduce_sum(self.user_seq, 1)  # [batch_size, fnum * eb_dim]
        inp = tf.concat([self.target_item, self.item_his_eb_sum, read_out, mean_memory * self.target_item], 1)
        self.build_fc_net(inp)
        self.build_logloss()

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.user_seq_ph: batch_data[0],
            self.user_seq_length_ph: batch_data[1],
            self.target_user_ph: batch_data[2],
            self.target_item_ph: batch_data[3],
            self.label_ph: batch_data[4],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: 0.8,
            self.batch_size: len(batch_data[0])
        })
        return loss

    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.user_seq_ph: batch_data[0],
            self.user_seq_length_ph: batch_data[1],
            self.target_user_ph: batch_data[2],
            self.target_item_ph: batch_data[3],
            self.label_ph: batch_data[4],
            self.reg_lambda: reg_lambda,
            self.keep_prob: 1.0,
            self.batch_size: len(batch_data[0])
        })
        return pred.reshape([-1, ]).tolist(), label.reshape([-1, ]).tolist(), loss

class DIEN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(DIEN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        mask = tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32)

        # 注意力 RNN 层
        with tf.name_scope('rnn_1'):
            user_seq_ht, _ = tf.nn.dynamic_rnn(
                GRUCell(hidden_size),
                inputs=self.user_seq,
                sequence_length=self.user_seq_length_ph,
                dtype=tf.float32,
                scope='gru1'
            )
        with tf.name_scope('attention'):
            atten_score, _ = self.attention(user_seq_ht, user_seq_ht, self.target_item, mask)
        with tf.name_scope('rnn_2'):
            _, seq_rep = dynamic_rnn(
                VecAttGRUCell(hidden_size),
                inputs=user_seq_ht,
                att_scores=atten_score,
                sequence_length=self.user_seq_length_ph,
                dtype=tf.float32,
                scope="argru1"
            )

        inp = tf.concat([seq_rep, self.target_user, self.target_item], axis=1)

        # 全连接层
        self.build_fc_net(inp)
        self.build_logloss()

    def attention(self, key, value, query, mask):
        # key: [B, T, Dk], query: [B, Dq], mask: [B, T]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        kq_inter = queries * key
        atten = tf.reduce_sum(kq_inter, axis=2)

        mask = tf.equal(mask, tf.ones_like(mask))  # [B, T]
        paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
        atten = tf.nn.softmax(tf.where(mask, atten, paddings))  # [B, T]
        atten = tf.expand_dims(atten, 2)

        res = tf.reduce_sum(atten * value, axis=1)
        return atten, res

class SASRec(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(SASRec, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        self.user_seq = self.multihead_attention(self.normalize(self.user_seq), self.user_seq)
        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, 1), [1, max_time_len, 1])
        self.target_item_t = tf.tile(tf.expand_dims(self.target_item, 1), [1, max_time_len, 1])

        self.mask = tf.expand_dims(tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32), axis=-1)
        self.mask_1 = tf.expand_dims(tf.sequence_mask(self.user_seq_length_ph - 1, max_time_len, dtype=tf.float32), axis=-1)
        self.get_mask = self.mask - self.mask_1
        self.seq_rep = self.user_seq * self.mask
        self.final_pred_rep = tf.reduce_sum(self.user_seq * self.get_mask, axis=1)

        # 序列的正负样本
        self.pos = self.user_seq[:, 1:, :]
        self.neg = self.user_seq[:, 2:, :]

        self.target_user_t = tf.tile(tf.expand_dims(self.target_user, 1), [1, max_time_len, 1])

        self.pos_seq_rep = tf.concat([self.seq_rep[:, 1:, :], self.pos, self.target_user_t[:, 1:, :]], axis=2)
        self.neg_seq_rep = tf.concat([self.seq_rep[:, 2:, :], self.neg, self.target_user_t[:, 2:, :]], axis=2)

        self.preds_pos = self.build_fc_net(self.pos_seq_rep)
        self.preds_neg = self.build_fc_net(self.neg_seq_rep)
        self.label_pos = tf.ones_like(self.preds_pos)
        self.label_neg = tf.zeros_like(self.preds_neg)

        self.loss = tf.losses.log_loss(self.label_pos, self.preds_pos) + tf.losses.log_loss(self.label_neg, self.preds_neg)

        # 针对目标用户和物品的预测
        inp = tf.concat([self.final_pred_rep, self.target_item, self.target_user], axis=1)
        self.y_pred = self.build_fc_net(inp)
        self.y_pred = tf.reshape(self.y_pred, [-1,])

        self.loss += tf.losses.log_loss(self.label_ph, self.y_pred)
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # 优化器和训练步骤
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_fc_net(self, inp):
        with tf.variable_scope('prediction_layer', reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense(inp, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 1, activation=tf.sigmoid, name='fc3')
        return fc3

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        '''应用多头注意力机制。

        Args:
            queries: 形状为 [N, T_q, C_q] 的 3D 张量。
            keys: 形状为 [N, T_k, C_k] 的 3D 张量。
            num_units: 一个标量，注意力的尺寸。
            num_heads: 一个整数，头的数量。
            scope: 可选的变量作用域。
            reuse: 布尔值，是否复用同一作用域内的变量。

        Returns:
            一个形状为 (N, T_q, C) 的 3D 张量。
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # 如果 num_units 未指定，默认使用 queries 的最后一个维度
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # 线性投影
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)     # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)     # (N, T_k, C)

            # 分割多头并合并
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # 矩阵乘法
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # 缩放
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # 关键字掩码
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])             # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2**32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # 激活
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # 查询掩码
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])              # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # (h*N, T_q, T_k)

            # Dropout
            outputs = tf.nn.dropout(outputs, self.keep_prob)

            # 加权求和
            outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)

            # 恢复形状
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # 残差连接
            outputs += queries

            # 归一化（如果需要，可以取消注释）
            # outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs

    def normalize(self,
                 inputs,
                 epsilon=1e-8,
                 scope="ln",
                 reuse=None):
        '''应用层归一化。

        Args:
            inputs: 一个 2 维或更高维度的张量，第一维为批量大小。
            epsilon: 一个小的浮点数，用于防止除零错误。
            scope: 可选的变量作用域。
            reuse: 布尔值，是否复用同一作用域内的变量。

        Returns:
            与输入形状和数据类型相同的张量。
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = gamma * normalized + beta

        return outputs

class SIMHard(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(SIMHard, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        
        # 商品类别ID在item_fnum中的索引为1
        item_id_index = 1

        # 从用户序列中获取商品ID部分，形状：[batch_size, max_time_len]
        user_item_ids = self.user_seq_ph[:, :, item_id_index]  # [B, T]
        
        # 获取目标商品的ID，形状：[batch_size]
        target_item_id = self.target_item_ph[:, item_id_index]  # [B]
        
        # 扩展目标商品ID的维度，便于比较
        target_item_id_expanded = tf.expand_dims(target_item_id, 1)  # [B, 1]
        # 将目标商品ID扩展到与用户序列相同的时间长度
        target_item_id_tiled = tf.tile(target_item_id_expanded, [1, max_time_len])  # [B, T]
        
        # 生成匹配掩码，标记用户序列中商品ID与目标商品ID相同的位置
        match_mask = tf.equal(user_item_ids, target_item_id_tiled)  # [B, T], dtype=bool

        # 获取用户序列的实际长度
        mask = tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.bool)  # [B, T]
        # 将匹配的掩码和实际序列长度的掩码进行逻辑与操作
        final_mask = tf.logical_and(match_mask, mask)  # [B, T]

        # 将匹配掩码转换为浮点型
        final_mask_float = tf.cast(final_mask, tf.float32)  # [B, T]
        # 扩展掩码维度，便于与嵌入序列相乘
        final_mask_expanded = tf.expand_dims(final_mask_float, -1)  # [B, T, 1]
        
        # 对用户序列嵌入进行掩码操作，保留匹配的部分
        user_seq_matching = self.user_seq * final_mask_expanded  # [B, T, item_fnum * eb_dim]

        # 使用目标注意力机制（与DIN中的attention相同)
        _, user_behavior_rep = self.attention(user_seq_matching, user_seq_matching, self.target_item, final_mask_float)

        # 将用户行为表示、目标用户和目标商品的嵌入拼接作为模型输入
        inp = tf.concat([user_behavior_rep, self.target_user, self.target_item], axis=1)
        
        # 全连接层和损失函数
        self.build_fc_net(inp)
        self.build_logloss()

    def attention(self, key, value, query, mask):
        # key: [B, T, Dk], query: [B, Dq], mask: [B, T]
        _, max_len, k_dim = key.get_shape().as_list()
        query_dense = tf.layers.dense(query, k_dim, activation=None)  # [B, Dk]
        queries = tf.tile(tf.expand_dims(query_dense, 1), [1, max_len, 1])  # [B, T, Dk]
        kq_inter = queries * key  # [B, T, Dk]
        atten = tf.reduce_sum(kq_inter, axis=2)  # [B, T]

        # 处理掩码，防止无效位置影响
        paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
        atten = tf.where(tf.equal(mask, 1.0), atten, paddings)  # [B, T]
        atten = tf.nn.softmax(atten)  # [B, T]

        atten_expanded = tf.expand_dims(atten, 2)  # [B, T, 1]
        res = tf.reduce_sum(atten_expanded * value, axis=1)  # [B, Dk]
        return atten, res

class TWIN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, top_k=50):
        super(TWIN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        
        # 获取用户行为序列的嵌入维度
        d_model = self.user_seq.get_shape().as_list()[-1]  # 应为 item_fnum * eb_dim
    
        # 对目标商品进行线性变换，以匹配用户行为序列的嵌入维度
        baseline_ads_emb = tf.layers.dense(self.target_item, d_model, activation=None, name='baseline_ads_emb')  # [B, D]
    
        # 计算用户序列中每个商品与目标商品的相似度得分（点积相似度）
        seq_score = tf.reduce_sum(self.user_seq * tf.expand_dims(baseline_ads_emb, 1), axis=-1)  # [B, T]
    
        # 获取序列掩码，处理不同长度的序列
        mask = tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32)  # [B, T]
    
        # 处理掩码，将无效位置的得分设置为一个较大的负值，避免干扰Top-K选择
        mask_score = tf.ones_like(seq_score) * (2 ** 31 - 1)
        seq_score = tf.where(mask > 0, seq_score, -mask_score)
    
        # 选取Top-K相似度最高的商品
        topk_score, topk_idx = tf.nn.top_k(seq_score, k=top_k)  # [B, K]
    
        # 获取Top-K商品的嵌入表示
        topk_emb = tf.batch_gather(self.user_seq, topk_idx)  # [B, K, D]
        # 获取Top-K商品的掩码
        topk_mask = tf.batch_gather(mask, topk_idx)  # [B, K]
    
        # 对Top-K商品的嵌入进行掩码处理，过滤掉无效的位置
        topk_emb = tf.multiply(topk_emb, tf.expand_dims(topk_mask, axis=2))  # [B, K, D]
    
        # 使用目标注意力机制
        user_behavior_rep, _ = self.attention_net_dot2(topk_emb, baseline_ads_emb, topk_mask, name='twin')
    
        # 将用户行为表示、目标用户和目标商品的嵌入拼接作为模型输入
        inp = tf.concat([user_behavior_rep, self.target_user, self.target_item], axis=1)
    
        # 构建全连接层和损失函数
        self.build_fc_net(inp)
        self.build_logloss()
    
    def attention_net_dot2(self, key, query, mask, name):
        # key: [B, K, Dk], query: [B, Dq], mask: [B, K]
        with tf.variable_scope(name):
            # 将query扩展并复制，以匹配key的维度
            queries = tf.expand_dims(query, 1)  # [B, 1, Dk]
            queries = tf.tile(queries, [1, tf.shape(key)[1], 1])  # [B, K, Dk]
    
            # 计算key和query的逐元素乘积（点积注意力）
            kq_inter = queries * key  # [B, K, Dk]
            atten = tf.reduce_sum(kq_inter, axis=2)  # [B, K]
    
            # 处理掩码，将无效位置的注意力权重设置为一个非常小的值，排除在softmax计算之外
            paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
            atten = tf.where(mask > 0, atten, paddings)
            atten = tf.nn.softmax(atten)  # [B, K]
    
            # 计算最终的注意力输出
            atten_expanded = tf.expand_dims(atten, 2)  # [B, K, 1]
            res = tf.reduce_sum(atten_expanded * key, axis=1)  # [B, Dk]
            return res, atten

class TWIN_V2(BaseModel):
    def __init__(self, feature_size, emb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, top_k=100, item_embeddings=None, n_clusters=10):
        super(TWIN_V2, self).__init__(feature_size, emb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)

        # 1. 商品嵌入聚类
        self.n_clusters = n_clusters
        self.item_embeddings = item_embeddings  # 预先计算的商品嵌入
        self.cluster_centers, self.cluster_labels = self.cluster_item_embeddings()

        # 2. 获取用户行为序列的嵌入维度
        d_model = self.user_seq.get_shape().as_list()[-1]  # 应为 item_fnum * emb_dim

        # 3. 目标商品嵌入
        target_item_emb = tf.layers.dense(self.target_item, d_model, activation=None, name='target_item_emb')  # [B, D]

        # 4. 计算用户行为序列中每个商品与目标商品的相似度得分（点积相似度）
        seq_score = tf.reduce_sum(self.user_seq * tf.expand_dims(target_item_emb, 1), axis=-1)  # [B, T]

        # 5. 获取序列掩码，处理不同长度的序列
        mask = tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32)  # [B, T]

        # 6. 处理掩码，设置无效位置得分为一个大的负值
        mask_score = tf.ones_like(seq_score) * (2 ** 31 - 1)
        seq_score = tf.where(mask > 0, seq_score, -mask_score)

        # 7. 选取Top-K相似度最高的商品
        topk_score, topk_idx = tf.nn.top_k(seq_score, k=top_k)  # [B, K]

        # 8. 获取Top-K商品的嵌入和掩码
        topk_emb = tf.batch_gather(self.user_seq, topk_idx)  # [B, K, D]
        topk_mask = tf.batch_gather(mask, topk_idx)  # [B, K]

        # 9. 对Top-K商品的嵌入进行掩码处理
        topk_emb = tf.multiply(topk_emb, tf.expand_dims(topk_mask, axis=2))  # [B, K, D]

        # 10. 使用聚类表示和集群感知的目标注意力机制
        user_behavior_rep, _ = self.cluster_aware_attention(topk_emb, target_item_emb, topk_mask)

        # 11. 将用户行为表示、目标用户和目标商品的嵌入拼接作为模型输入
        inp = tf.concat([user_behavior_rep, self.target_user, self.target_item], axis=1)

        # 12. 构建全连接层和损失函数
        self.build_fc_net(inp)
        self.build_logloss()

    def cluster_item_embeddings(self):
        """对商品嵌入进行聚类"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.item_embeddings)
        return kmeans.cluster_centers_, kmeans.labels_

    def cluster_aware_attention(self, cluster_key, query, mask):
        """实现集群感知的目标注意力机制"""
        # cluster_key: [B, K, Dk], query: [B, Dq], mask: [B, K]
        with tf.variable_scope('cluster_aware_attention'):
            queries = tf.expand_dims(query, 1)  # [B, 1, Dk]
            queries = tf.tile(queries, [1, tf.shape(cluster_key)[1], 1])  # [B, K, Dk]

            # 计算点积注意力
            kq_inter = queries * cluster_key  # [B, K, Dk]
            atten = tf.reduce_sum(kq_inter, axis=2)  # [B, K]

            # 处理掩码，将无效位置的注意力权重设置为一个很小的值
            paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
            atten = tf.where(mask > 0, atten, paddings)
            atten = tf.nn.softmax(atten)  # [B, K]

            # 计算最终的聚合输出
            atten_expanded = tf.expand_dims(atten, 2)  # [B, K, 1]
            weighted_output = tf.reduce_sum(atten_expanded * cluster_key, axis=1)  # [B, Dk]
            return weighted_output, atten



class ETA(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, codelen=8, top_k=100):
        super(ETA, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        
        # 设置参数
        self.codelen = codelen
        self.top_k = top_k
        
        # 获取嵌入维度
        d_model = self.user_seq.get_shape().as_list()[-1]  # D = item_fnum * eb_dim

        # 对目标商品进行线性变换
        baseline_ads_emb = tf.layers.dense(self.target_item, d_model, name='baseline_ads_emb')  # [B, D]

        # 定义随机旋转矩阵，用于LSH近似最近邻搜索
        random_rotations = tf.get_variable("random_rotations", shape=[d_model, self.codelen], trainable=False, initializer=tf.glorot_uniform_initializer())

        # 调用 topk_retrieval 函数，找到与目标商品最相似的 Top-K 用户历史行为
        topk_emb, topk_mask, topk_indices = self.topk_retrieval(random_rotations, baseline_ads_emb, self.user_seq, self.user_seq_length_ph, top_k=self.top_k, scope='topk_retrieval')  # [B, K, D], [B, K], [B, K]

        # 根据掩码处理 Top-K 嵌入
        topk_emb = tf.multiply(topk_emb, tf.expand_dims(topk_mask, axis=2))  # [B, K, D]

        # 使用非线性转换进行注意力计算
        user_behavior_rep = self.listseq_attention_new(topk_emb, baseline_ads_emb, trans_type="nonlinear", name='eta')  # [B, D]

        # 拼接最终输入
        inp = tf.concat([user_behavior_rep, self.target_user, self.target_item], axis=1)

        # 构建全连接层和损失函数
        self.build_fc_net(inp)
        self.build_logloss()
    
    def topk_retrieval(self, random_rotations, query_emb, key_emb_seq, seq_length, top_k=100, scope='topk_retrieval'):
        with tf.variable_scope(scope):
            # 计算查询向量的哈希码
            query_code = tf.sign(tf.matmul(query_emb, random_rotations))  # [B, codelen]

            # 将用户历史行为序列展平成二维
            batch_size = tf.shape(query_emb)[0]
            max_time_len = tf.shape(key_emb_seq)[1]
            d_model = key_emb_seq.get_shape().as_list()[-1]
            flattened_key_emb = tf.reshape(key_emb_seq, [batch_size * max_time_len, d_model])  # [B*T, D]

            # 计算历史行为的哈希码
            key_codes = tf.sign(tf.matmul(flattened_key_emb, random_rotations))  # [B*T, codelen]

            # 计算哈希码之间的距离（汉明距离）
            key_codes_reshaped = tf.reshape(key_codes, [batch_size, max_time_len, self.codelen])  # [B, T, codelen]
            hamming_distance = tf.reduce_sum(tf.cast(tf.not_equal(tf.expand_dims(query_code, 1), key_codes_reshaped), tf.float32), axis=2)  # [B, T]

            # 使用掩码，防止填充位置干扰
            mask = tf.sequence_mask(seq_length, maxlen=max_time_len, dtype=tf.float32)  # [B, T]
            paddings = tf.ones_like(hamming_distance) * 1e9  # 一个大的值
            hamming_distance = tf.where(tf.equal(mask, 1.0), hamming_distance, paddings)

            # 选取哈希距离最小的 Top-K 项
            _, topk_indices = tf.nn.top_k(-hamming_distance, k=top_k)  # [B, K]

            # 获取 Top-K 的嵌入和掩码
            batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, top_k])  # [B, K]
            gather_indices = tf.stack([batch_indices, topk_indices], axis=2)  # [B, K, 2]
            topk_emb = tf.gather_nd(key_emb_seq, gather_indices)  # [B, K, D]
            topk_mask = tf.gather_nd(mask, gather_indices)  # [B, K]

            return topk_emb, topk_mask, topk_indices

    def listseq_attention_new(self, key, query, trans_type="nonlinear", name='eta'):
        with tf.variable_scope(name):
            # key: [B, K, D], query: [B, D]
            if trans_type == "nonlinear":
                # 对 key 和 query 进行非线性转换
                D = key.get_shape().as_list()[-1]
                key_trans = tf.layers.dense(key, D, activation=tf.nn.relu, name='key_transform')  # [B, K, D]
                query_trans = tf.layers.dense(query, D, activation=tf.nn.relu, name='query_transform')  # [B, D]
            else:
                key_trans = key
                query_trans = query

            # 计算注意力权重
            queries = tf.expand_dims(query_trans, 1)  # [B, 1, D]
            scores = tf.reduce_sum(key_trans * queries, axis=2)  # [B, K]
            attention_weights = tf.nn.softmax(scores)  # [B, K]
            attention_weights = tf.expand_dims(attention_weights, 2)  # [B, K, 1]

            # 加权求和得到用户行为表示
            user_behavior_rep = tf.reduce_sum(attention_weights * key, axis=1)  # [B, D]
            return user_behavior_rep

class SDIM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(SDIM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        
        # Set constants
        codelen = 4
        num_hashes = 1
        powers_of_two = tf.constant([2.0 ** i for i in range(codelen)], dtype=tf.float32)  # Shape [codelen]
        
        # Get dimensions
        d_model = self.user_seq.get_shape().as_list()[-1]  # item_fnum * eb_dim
        
        # Compute mean pooling of user sequence embeddings
        all_sum = tf.reduce_mean(self.user_seq, axis=1)  # Shape [batch_size, d_model]
        
        # Map target item embedding to the same dimension if needed
        baseline_ads_emb = tf.layers.dense(self.target_item, d_model, name='baseline_ads_emb')  # Shape [batch_size, d_model]
        
        # Random rotations for simhash
        random_rotations = tf.get_variable("random_rotations", shape=[d_model, num_hashes, codelen],
                                           trainable=False, initializer=tf.keras.initializers.glorot_uniform())
        
        # Create mask for sequence lengths
        mask = tf.sequence_mask(self.user_seq_length_ph, max_time_len, dtype=tf.float32)  # Shape [batch_size, max_time_len]
        
        # Get collision mask
        collide_mask = self.topk_retrieval_SDIM(random_rotations, baseline_ads_emb, self.user_seq, mask, powers_of_two,
                                                scope='topk_retrieval_SDIM')  # Shape [batch_size, num_hashes, max_time_len]
        
        # Compute collision embeddings
        user_seq_expanded = tf.expand_dims(self.user_seq, axis=1)  # Shape [batch_size, 1, max_time_len, d_model]
        collide_mask_expanded = tf.expand_dims(collide_mask, axis=3)  # Shape [batch_size, num_hashes, max_time_len, 1]
        collide_embs = tf.multiply(user_seq_expanded, collide_mask_expanded)  # Shape [batch_size, num_hashes, max_time_len, d_model]
        collide_embs = tf.reduce_sum(collide_embs, axis=2)  # Shape [batch_size, num_hashes, d_model]
        
        # If num_hashes == 1, squeeze the dimension
        collide_embs = tf.squeeze(collide_embs, axis=1)  # Shape [batch_size, d_model]
        
        # Perform attention between collided embeddings and baseline ads embedding
        _, long_interest = self.attention(tf.expand_dims(collide_embs, axis=1),
                                          tf.expand_dims(collide_embs, axis=1),
                                          baseline_ads_emb,
                                          mask=None)
        
        # Concatenate features
        inp = tf.concat([all_sum, long_interest, self.target_user, self.target_item], axis=1)
        
        # Build fully connected layers
        self.build_fc_net(inp)
        self.build_logloss()

    def topk_retrieval_SDIM(self, random_rotations, target_item, history_sequence, mask, powers_of_two, scope='topk_retrieval_SDIM'):
        with tf.variable_scope(name_or_scope=scope, default_name="topk_retrieval_SDIM"):
            # Compute hash codes
            target_hash = self.simhash_SDIM(tf.expand_dims(target_item, axis=1), random_rotations, powers_of_two)  # [batch_size, 1, num_hashes]
            sequence_hash = self.simhash_SDIM(history_sequence, random_rotations, powers_of_two)  # [batch_size, max_time_len, num_hashes]
            
            # Compute collision mask
            bucket_match = tf.transpose(target_hash - sequence_hash, perm=[0, 2, 1])  # [batch_size, num_hashes, max_time_len]
            collide_mask = tf.cast(tf.equal(bucket_match, 0), tf.float32)  # [batch_size, num_hashes, max_time_len]
            
            # Apply sequence mask
            if mask is not None:
                mask_expanded = tf.expand_dims(mask, axis=1)  # [batch_size, 1, max_time_len]
                collide_mask *= mask_expanded  # Element-wise multiplication
            return collide_mask

    def simhash_SDIM(self, x, random_rotations, powers_of_two, scope='simhash_SDIM'):
        with tf.variable_scope(name_or_scope=scope, default_name="simhash_SDIM"):
            # Apply random rotations
            rotated_vecs = tf.einsum('btd,dnc->btnc', x, random_rotations)  # [batch_size, time_len, num_hashes, codelen]
            
            # Compute hash codes
            hash_code = tf.nn.relu(tf.sign(rotated_vecs))  # [batch_size, time_len, num_hashes, codelen]
            hash_code = tf.cast(hash_code, tf.float32)
            
            # Convert binary hash code to integer hash buckets
            powers_of_two_expanded = tf.reshape(powers_of_two, [1, 1, 1, -1])  # [1, 1, 1, codelen]
            hash_code *= powers_of_two_expanded  # Element-wise multiplication
            hash_bucket = tf.reduce_sum(hash_code, axis=-1)  # [batch_size, time_len, num_hashes]
            return hash_bucket

    def attention(self, key, value, query, mask):
        # key: [batch_size, T, D]
        # query: [batch_size, D]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [batch_size, T, D]
        kq_inter = queries * key  # Element-wise multiplication
        atten = tf.reduce_sum(kq_inter, axis=2)  # [batch_size, T]

        # Apply softmax
        if mask is not None:
            mask = tf.equal(mask, tf.ones_like(mask))  # [batch_size, T]
            paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
            atten = tf.nn.softmax(tf.where(mask, atten, paddings))  # [batch_size, T]
        else:
            atten = tf.nn.softmax(atten)  # [batch_size, T]
        atten = tf.expand_dims(atten, 2)  # [batch_size, T, 1]

        # Compute weighted sum
        res = tf.reduce_sum(atten * value, axis=1)  # [batch_size, D]
        return atten, res

class DGIN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer):
        super(DGIN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
        
        # 分组模块
        with tf.name_scope('group_module'):
            # 使用 item_id 进行分组
            self.grouped_behaviors = self.group_by_item_id(self.user_seq)
            
            # 对组内行为进行 self-attention
            self.group_representations = self.apply_self_attention(self.grouped_behaviors)
        
        # 目标模块
        with tf.name_scope('target_module'):
            # 对组间行为和目标项目进行 target attention
            self.final_interest_representation = self.apply_target_attention(self.group_representations, self.target_item)
        
        # 全连接层
        inp = tf.concat([self.final_interest_representation, self.target_user, self.target_item], axis=1)
        self.build_fc_net(inp)
        self.build_logloss()

    def group_by_item_id(self, user_seq):
        # 假设 user_seq 的每一行为包含 item_id 信息
        # 这里实现简单的分组逻辑，可以根据实际需求进行调整
        # 返回值为分组后的行为序列
        grouped_behaviors = {}  # 用字典存储分组结果，key 是 item_id，value 是行为列表
        for behavior in user_seq:
            item_id = behavior  # 提取每个的item_id
            if item_id not in grouped_behaviors:
                grouped_behaviors[item_id] = []
            grouped_behaviors[item_id].append(behavior) # 这个地方有问题！！！
        
        # 将字典转换为列表形式
        grouped_list = [group for group in grouped_behaviors.values()]
        return grouped_list

    def apply_self_attention(self, grouped_behaviors):
        # 对每个组应用 self-attention
        group_representations = []
        for group in grouped_behaviors:
            group_rep = self.multihead_self_attention(group)
            group_representations.append(group_rep)
        return group_representations

    def apply_target_attention(self, group_representations, target_item):
        # 对组间表示和目标项目应用 target attention
        # 使用 DIN 中的 attention 函数实现
        _, target_representation = self.attention(group_representations, group_representations, target_item, mask=None)
        return target_representation

    def multihead_self_attention(self, group):
        # 使用多头自注意力机制计算组的表示
        # 经典的 QKV 范式实现
        num_units = group.get_shape().as_list()[-1]
        
        Q = tf.layers.dense(group, num_units, activation=None)
        K = tf.layers.dense(group, num_units, activation=None)
        V = tf.layers.dense(group, num_units, activation=None)

        # 计算注意力权重
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
        
        # softmax 激活
        outputs = tf.nn.softmax(outputs)
        
        # 加权求和
        outputs = tf.matmul(outputs, V)
        return outputs

    def attention(self, key, value, query, mask):
        # key: [B, T, Dk], query: [B, Dq], mask: [B, T]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        kq_inter = queries * key
        atten = tf.reduce_sum(kq_inter, axis=2)

        if mask is not None:
            mask = tf.equal(mask, tf.ones_like(mask))  # [B, T]
            paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
            atten = tf.nn.softmax(tf.where(mask, atten, paddings))  # [B, T]
        else:
            atten = tf.nn.softmax(atten)  # [B, T]
        atten = tf.expand_dims(atten, 2)

        res = tf.reduce_sum(atten * value, axis=1)
        return atten, res
