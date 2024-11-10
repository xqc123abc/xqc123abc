import os
import sys
from dataloader import *
from baselines import *
from rec import *
from sklearn.metrics import *
import random
import time
import numpy as np
import pickle as pkl
import math
from collections import defaultdict

random.seed(1111)

EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16 * 2
EVAL_BATCH_SIZE = 500

# for TMALL
FEAT_SIZE_TMALL = 1529672 + 6 #(6 is for time context)
DATA_DIR_TMALL = '../data/tmall/feateng_data/'
MAX_LEN_TMALL = 20

# for Taobao
FEAT_SIZE_TAOBAO = 5062314
DATA_DIR_TAOBAO = '../data/taobao/feateng_data/'
MAX_LEN_TAOBAO = 500

# for Alipay
FEAT_SIZE_ALIPAY = 2836410
DATA_DIR_ALIPAY = '../data/alipay/feateng_data/'
MAX_LEN_ALIPAY = 12


def restore(data_set_name, target_test_file, user_seq_file, user_feat_dict_file, item_feat_dict_file,
            model_type, train_batch_size, feature_size, eb_dim, hidden_size, max_time_len, 
            lr, reg_lambda, user_fnum, item_fnum, emb_initializer):
    print('restore begin')
    tf.reset_default_graph()
    if model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'Caser': 
        model = Caser(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer) 
    elif model_type == 'DIN': 
        model = DIN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer) 
    elif model_type == 'DIEN': 
        model = DIEN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer) 
    elif model_type == 'SASRec': 
        model = SASRec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'MIMN': 
        model = MIMN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'HPMN': 
        model = HPMN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'SIMHard':
        model = SIMHard(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'TWIN':
        model = TWIN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'TWIN_V2':
        item_emb = train_din_and_get_item_embeddings(dataset_size, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, lr, reg_lambda, train_batch_size, user_seq_file, target_train_file, user_feat_dict_file, item_feat_dict_file)
        model = TWIN_V2(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, top_k=100, item_embeddings=item_emb, n_clusters=10)
    elif  model_type == 'ETA':
       model = ETA(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'SDIM':
        model = SDIM(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'DGIN':
        model = DGIN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/{}/ckpt'.format(data_set_name, max_time_len, model_name))
        print('restore eval begin')
        _, logloss, rig, auc, gauc = eval(model, sess, target_test_file, user_seq_file, user_feat_dict_file, item_feat_dict_file, max_time_len, reg_lambda)

        print('RESTORE, LOGLOSS %.4f  RIG: %.4f  AUC: %.4f  GAUC:%.4f' % (logloss, rig, auc, gauc))
        with open('logs_{}/{}/{}.txt'.format(data_set_name, max_time_len, model_type), 'a') as f:
            results = [train_batch_size, lr, reg_lambda, logloss, rig, auc]
            results = [model_type] + [str(res) for res in results]
            result_line = '\t'.join(results) + '\n'
            f.write(result_line)
        

def eval(model, sess, target_file, user_seq_file, user_feat_dict_file, item_feat_dict_file, max_time_len, reg_lambda):
    preds = []
    labels = []
    losses = []
    user_ids = []  # 用于存储用户ID

    data_loader = DataLoader(EVAL_BATCH_SIZE, user_seq_file, target_file, user_feat_dict_file, item_feat_dict_file, max_time_len)
    
    t = time.time()
    for batch_data in data_loader:
        pred, label, loss = model.eval(sess, batch_data, reg_lambda)
        preds += pred
        labels += label
        user_ids += [user_id[0] for user_id in batch_data[2]]
        losses.append(loss)

    logloss = log_loss(labels, preds)
    rig = 1 - (logloss / -(0.5 * math.log(0.5) + (1 - 0.5) * math.log(1 - 0.5)))
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)
    
    # 计算GAUC
    user_auc_dict = defaultdict(list)
    for user_id, pred, label in zip(user_ids, preds, labels):
        user_auc_dict[user_id].append((pred, label))

    # 计算每个用户的AUC
    total_auc = 0
    total_clicks = 0
    for user_id, user_data in user_auc_dict.items():
        user_preds, user_labels = zip(*user_data)
        if len(set(user_labels)) > 1:  # 确保有正负样本
            user_auc = roc_auc_score(user_labels, user_preds)
            # 仅计算有点击行为（正样本）的权重
            num_clicks = sum(user_labels)
            total_auc += user_auc * num_clicks
            total_clicks += num_clicks
    #  计算GAUC
    gauc = total_auc / total_clicks if total_clicks > 0 else 0
    print("EVAL TIME: %.4fs" % (time.time() - t))
    print("GAUC: %.4f" % gauc)
    return loss, logloss, rig, auc, gauc


def train(data_set_name, target_train_file, target_vali_file, user_seq_file, user_feat_dict_file, item_feat_dict_file, 
        model_type, train_batch_size, feature_size, eb_dim, hidden_size, max_time_len, lr, reg_lambda, dataset_size, 
        user_fnum, item_fnum, emb_initializer):
    tf.reset_default_graph()
    
    if model_type == 'GRU4Rec':
        model = GRU4Rec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'Caser': 
        model = Caser(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'DIN': 
        model = DIN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)    
    elif model_type == 'DIEN': 
        model = DIEN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer) 
    elif model_type == 'SASRec': 
        model = SASRec(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'MIMN': 
        model = MIMN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'HPMN': 
        model = HPMN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'SIMHard':
        model = SIMHard(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'TWIN':
        model = TWIN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'TWIN_V2':
        item_emb = train_din_and_get_item_embeddings(dataset_size, feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, lr, reg_lambda, train_batch_size, user_seq_file, target_train_file, user_feat_dict_file, item_feat_dict_file)
        model = TWIN_V2(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer, top_k=100, item_embeddings=item_emb, n_clusters=10)
    elif model_type == 'ETA':
        model = ETA(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'SDIM':
        model = SDIM(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    elif model_type == 'DGIN':
        model = DGIN(feature_size, eb_dim, hidden_size, max_time_len, user_fnum, item_fnum, emb_initializer)
    else:
        print('WRONG MODEL TYPE')
        exit(1)

    training_monitor = {
        'train_loss' : [],
        'vali_loss' : [],
        'logloss' : [],
        'rig' : [],
        'auc' : [],
        'gauc' : []
    }

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process
        step = 0
        vali_loss, logloss, rig, auc, gauc = eval(model, sess, target_vali_file, user_seq_file, user_feat_dict_file, item_feat_dict_file, max_time_len, reg_lambda)
        
        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(vali_loss)
        training_monitor['logloss'].append(logloss)
        training_monitor['rig'].append(rig)
        training_monitor['auc'].append(auc)

        print("STEP %d  LOSS TRAIN: NULL  LOSS VALI: %.4f  LOGLOSS: %.4f  RIG: %.4f  AUC: %.4f  GAUC: %.4f" % (step, vali_loss, logloss, rig, auc, gauc))
        early_stop = False
        eval_iter_num = (dataset_size // 5) // train_batch_size
        # begin training process
        for epoch in range(1):
            if early_stop:
                break
            data_loader = DataLoader(train_batch_size, user_seq_file, target_train_file, user_feat_dict_file, item_feat_dict_file, max_time_len)
            
            for batch_data in data_loader:
                # print(batch_data)
                if early_stop:
                    break
                loss = model.train(sess, batch_data, lr, reg_lambda)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []
                    
                    vali_loss, logloss, rig, auc, gauc = eval(model, sess, target_vali_file, user_seq_file, user_feat_dict_file, item_feat_dict_file, max_time_len, reg_lambda)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['logloss'].append(logloss)
                    training_monitor['rig'].append(rig)
                    training_monitor['auc'].append(auc)
                    training_monitor['gauc'].append(gauc)
                    print("STEP %d  LOSS TRAIN: %.4f  LOSS VALI: %.4f  LOGLOSS: %.4f  RIG: %.4f  AUC: %.4f  GAUC:%.4f" % (step, train_loss, vali_loss, logloss, rig, auc, gauc))
                    if training_monitor['auc'][-1] > max(training_monitor['auc'][:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name)):
                            os.makedirs('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name))
                        save_path = 'save_model_{}/{}/{}/ckpt'.format(data_set_name, max_time_len, model_name)
                        model.save(sess, save_path)

                    if len(training_monitor['vali_loss']) > 2 and epoch > 0:
                        if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                            early_stop = True
                        if (training_monitor['vali_loss'][-2] - training_monitor['vali_loss'][-1]) <= 0.001 and (training_monitor['vali_loss'][-3] - training_monitor['vali_loss'][-2]) <= 0.001:
                            early_stop = True

        # generate log
        if not os.path.exists('logs_{}/{}/'.format(data_set_name, max_time_len)):
            os.makedirs('logs_{}/{}/'.format(data_set_name, max_time_len))
        model_name = '{}_{}_{}_{}'.format(model_type, train_batch_size, lr, reg_lambda)

        with open('logs_{}/{}/{}.pkl'.format(data_set_name, max_time_len, model_name), 'wb') as f:
            pkl.dump(training_monitor, f)

def list_available_devices():
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        devices = sess.list_devices()
        print("可用的设备列表：")
        for device in devices:
            print(device)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("PLEASE INPUT [MODEL TYPE] [GPU] [DATASET]")
        sys.exit(0)
    model_type = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    data_set_name = sys.argv[3]
    
    list_available_devices()

    if data_set_name == 'tmall':
        user_fnum = 3 
        item_fnum = 4
        target_train_file = DATA_DIR_TMALL + 'target_train.txt'
        target_vali_file = DATA_DIR_TMALL + 'target_vali.txt'
        target_test_file = DATA_DIR_TMALL + 'target_test.txt'
        user_seq_file = DATA_DIR_TMALL + 'user_seq.txt'

        user_feat_dict_file = DATA_DIR_TMALL + 'user_feat_dict.pkl'
        item_feat_dict_file = DATA_DIR_TMALL + 'item_feat_dict.pkl'
        
        # model parameter
        feature_size = FEAT_SIZE_TMALL
        max_time_len = MAX_LEN_TMALL #100
        dataset_size = 847568

        emb_initializer = None

    elif data_set_name == 'taobao':
        user_fnum = 1 
        item_fnum = 2

        target_train_file = DATA_DIR_TAOBAO + 'target_train.txt'
        target_vali_file = DATA_DIR_TAOBAO + 'target_vali.txt'
        target_test_file = DATA_DIR_TAOBAO + 'target_test.txt'
        user_seq_file = DATA_DIR_TAOBAO + 'user_seq.txt'

        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_TAOBAO + 'item_feat_dict.pkl'
        
        # model parameter
        feature_size = FEAT_SIZE_TAOBAO
        max_time_len = MAX_LEN_TAOBAO #100
        dataset_size = 1962046

        emb_initializer = None
    elif data_set_name == 'alipay':
        user_fnum = 1
        item_fnum = 3

        target_train_file = DATA_DIR_ALIPAY + 'target_train.txt'
        target_vali_file = DATA_DIR_ALIPAY + 'target_vali.txt'
        target_test_file = DATA_DIR_ALIPAY + 'target_test.txt'
        user_seq_file = DATA_DIR_ALIPAY + 'user_seq.txt'

        user_feat_dict_file = None
        item_feat_dict_file = DATA_DIR_ALIPAY + 'item_feat_dict.pkl'
        
        # model parameter
        feature_size = FEAT_SIZE_ALIPAY
        max_time_len = MAX_LEN_ALIPAY #60
        dataset_size = 996616

        emb_initializer = None     

    else:
        print('WRONG DATASET NAME: {}'.format(data_set_name))
        exit()

    ################################## training hyper params ##################################
    reg_lambdas = [1e-4, 5e-4]
    hyper_paras = [(50, 5e-4), (100, 1e-3)]

    for hyper in hyper_paras:
        train_batch_size, lr = hyper
        for reg_lambda in reg_lambdas:
            train(data_set_name, target_train_file, target_vali_file, user_seq_file, user_feat_dict_file, item_feat_dict_file, 
                    model_type, train_batch_size, feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, lr, reg_lambda, dataset_size, 
                    user_fnum, item_fnum, emb_initializer)
            restore(data_set_name, target_test_file, user_seq_file, user_feat_dict_file, item_feat_dict_file, model_type, train_batch_size, 
                feature_size, EMBEDDING_SIZE, HIDDEN_SIZE, max_time_len, lr, reg_lambda, user_fnum, item_fnum, emb_initializer)