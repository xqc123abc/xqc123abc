import tensorflow as tf
import numpy as np

class RecBase(object):
    def __init__(self, feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer):
        # 输入占位符
        with tf.name_scope('rec/inputs'):
            self.seq_ph = tf.placeholder(tf.int32, [None, b_num, record_fnum], name='seq_ph')
            self.seq_length_ph = tf.placeholder(tf.int32, [None,], name='seq_length_ph')
            self.target_ph = tf.placeholder(tf.int32, [None, record_fnum], name='target_ph')
            self.label_ph = tf.placeholder(tf.int32, [None,], name='label_ph')

            # 学习率
            self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
            # 正则化系数
            self.reg_lambda = tf.placeholder(tf.float32, [], name='reg_lambda')
            # dropout 保留率
            self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        
        # 词嵌入
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            if emb_initializer is not None:
                self.emb_mtx = tf.get_variable('emb_mtx', initializer=emb_initializer)
            else:
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim], initializer=tf.truncated_normal_initializer())
                self.emb_mtx_mask = tf.constant(value=1.0, shape=[feature_size - 1, eb_dim])
                self.emb_mtx_mask = tf.concat([tf.constant(value=0.0, shape=[1, eb_dim]), self.emb_mtx_mask], axis=0)
                self.emb_mtx = self.emb_mtx * self.emb_mtx_mask

        self.seq = tf.nn.embedding_lookup(self.emb_mtx, self.seq_ph)  # [B, b_num, record_fnum, eb_dim]
        self.seq = tf.reshape(self.seq, [-1, b_num, record_fnum * eb_dim])  # [B, b_num, record_fnum * eb_dim]
        self.target = tf.nn.embedding_lookup(self.emb_mtx, self.target_ph)  # [B, record_fnum, eb_dim]
        self.target = tf.reshape(self.target, [-1, record_fnum * eb_dim])  # [B, record_fnum * eb_dim]


    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='rec_bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='rec_fc1')
        dp1 = tf.nn.dropout(fc1, keep_prob=self.keep_prob, name='rec_dp1')  # dropout 保留率
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='rec_fc2')
        dp2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob, name='rec_dp2')
        fc3 = tf.layers.dense(dp2, 2, activation=None, name='rec_fc3')
        score = tf.nn.softmax(fc3, axis=1)
        # 输出预测值
        self.y_pred = tf.reshape(score[:, 0], [-1,])  # [B,]


    def build_reward(self):
        # 使用 RIG 作为奖励信号
        self.ground_truth = tf.cast(self.label_ph, tf.float32)  # [B,]
        # 计算奖励
        self.reward = self.ground_truth * tf.log(tf.clip_by_value(self.y_pred, 1e-10, 1.0)) + \
                      (1.0 - self.ground_truth) * tf.log(tf.clip_by_value(1.0 - self.y_pred, 1e-10, 1.0))
        self.reward = 1.0 - (self.reward / tf.log(0.5))  # 使用 RIG 作为奖励信号
        self.edge = -tf.ones_like(self.reward)  # [B,]
        # 如果 reward 小于 -1，则设为 -1
        self.reward = tf.where(self.reward < -1.0, self.edge, self.reward)  # [B,]


    def build_logloss(self):
        # 损失函数
        self.log_loss = tf.losses.log_loss(labels=self.label_ph, predictions=self.y_pred)
        self.loss = self.log_loss
        # 添加正则化项
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)


    def build_optimizer(self):    
        # 优化器和训练步骤
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='rec_optimizer')
        self.train_step = self.optimizer.minimize(self.loss)


    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.seq_ph: batch_data[0],
            self.seq_length_ph: batch_data[1],
            self.target_ph: batch_data[2],
            self.label_ph: batch_data[3],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: 0.8
        })
        return loss


    def eval(self, sess, batch_data, reg_lambda):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.seq_ph: batch_data[0],
            self.seq_length_ph: batch_data[1],
            self.target_ph: batch_data[2],
            self.label_ph: batch_data[3],
            self.reg_lambda: reg_lambda,
            self.keep_prob: 1.0
        })
        
        return pred.reshape([-1,]).tolist(), label.reshape([-1,]).tolist(), loss


    def get_reward(self, sess, batch_data):
        reward = sess.run(self.reward, feed_dict={
            self.seq_ph: batch_data[0],
            self.seq_length_ph: batch_data[1],
            self.target_ph: batch_data[2],
            self.label_ph: batch_data[3],
            self.keep_prob: 1.0
        })
        return np.reshape(reward, [-1, 1])  # [B,1]
    
        
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)


    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('模型已从 {} 恢复'.format(path))


class RecSum(RecBase):
    def __init__(self, feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer):
        super(RecSum, self).__init__(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)

        # 使用 sum pooling 来建模用户行为，填充为零（嵌入 ID 也是零）
        user_behavior_rep = tf.reduce_sum(self.seq, axis=1)  # [B, record_fnum * eb_dim]
        
        inp = tf.concat([user_behavior_rep, self.target], axis=1)  # [B, record_fnum * eb_dim * 2]

        # 全连接层
        self.build_fc_net(inp)
        self.build_reward()
        self.build_logloss()
        self.build_optimizer()


class RecAtt(RecBase):
    def __init__(self, feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer):
        super(RecAtt, self).__init__(feature_size, eb_dim, hidden_size, b_num, record_fnum, emb_initializer)
        mask = tf.sequence_mask(self.seq_length_ph, b_num, dtype=tf.float32)  # [B, b_num]
        self.atten, user_behavior_rep = self.attention(self.seq, self.seq, self.target, mask)  # [B, b_num, ...], [B, ...]
        self.atten = tf.reshape(self.atten, [-1, b_num])  # [B, b_num]
        inp = tf.concat([user_behavior_rep, self.target], axis=1)  # [B, ...]

        # 全连接层
        self.build_fc_net(inp)
        self.build_reward()
        self.build_logloss()
        self.build_optimizer()


    def attention(self, key, value, query, mask):
        # key: [B, T, Dk], value: [B, T, Dv], query: [B, Dq], mask: [B, T]
        _, max_len, k_dim = key.get_shape().as_list()
        query = tf.layers.dense(query, k_dim, activation=None, name='att_query')  # [B, Dk]
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        kq_inter = queries * key  # [B, T, Dk]
        atten = tf.reduce_sum(kq_inter, axis=2)  # [B, T]
        
        mask = tf.equal(mask, tf.ones_like(mask))  # [B, T]
        paddings = tf.ones_like(atten) * (-2 ** 32 + 1)  # [B, T]
        atten = tf.nn.softmax(tf.where(mask, atten, paddings))  # [B, T]
        atten = tf.expand_dims(atten, 2)  # [B, T, 1]

        res = tf.reduce_sum(atten * value, axis=1)  # [B, Dv]
        return atten, res