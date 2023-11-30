''' Module for training TF parts.'''

import pandas as pd
import sys
import numpy as np
import tensorflow.compat.v1 as tf
import time
from os.path import join
from src import param
from src.batchloader import BatchLoader
from src.utils import vec_length, ModelList
from src.model import UKGE_LOGI, UKGE_RECT
from src.validator import UKGE_LOGI_VALIDATOR, UKGE_RECT_VALIDATOR
tf.disable_v2_behavior()

class Trainer(object):
    def __init__(self):
        self.batch_size = 128
        self.dim = 64
        self.this_data = None
        self.tf_parts = None
        self.save_path = 'this-distmult.ckpt'
        self.data_save_path = 'this-data.bin'
        self.file_val = ""
        self.L1 = False

    def build(self, data_obj, save_dir,
              model_save='model.bin',
              data_save='data.bin'):
        """
        All files are stored in save_dir.
        output files:
        1. tf model
        2. this_data (Data())
        3. training_loss.csv, val_loss.csv
        :param model_save: filename for model
        :param data_save: filename for self.this_data
        :param knn_neg: use kNN negative sampling
        :return:
        """
        self.verbose = param.verbose  # print extra information
        self.this_data = data_obj
        self.dim = self.this_data.dim = param.dim
        self.batch_size = self.this_data.batch_size = param.batch_size
        self.neg_per_positive = param.neg_per_pos
        self.reg_scale = param.reg_scale

        self.batchloader = BatchLoader(self.this_data, self.batch_size, self.neg_per_positive)

        self.p_neg = param.p_neg
        self.p_psl = param.p_psl

        # paths for saving
        self.save_dir = save_dir
        self.save_path = join(save_dir, model_save)  # tf model
        self.data_save_path = join(save_dir, data_save)  # this_data (Data())
        self.train_loss_path = join(save_dir, 'training_loss.csv')
        self.val_loss_path = join(save_dir, 'val_loss.csv')

        print('Now using model: ', param.model)

        self.model = param.model
        if self.model == ModelList.LOGI:
            self.tf_parts = UKGE_LOGI(
                num_rels=self.this_data.num_rels(),
                num_cons=self.this_data.num_cons(),
                dim=self.dim,
                batch_size=self.batch_size,
                neg_per_positive=self.neg_per_positive, 
                p_neg=self.p_neg
            )
            self.validator = UKGE_LOGI_VALIDATOR()
        elif self.model == ModelList.RECT:
            self.tf_parts = UKGE_RECT(
                num_rels=self.this_data.num_rels(),
                num_cons=self.this_data.num_cons(),
                dim=self.dim,
                batch_size=self.batch_size,
                neg_per_positive=self.neg_per_positive, 
                p_neg=self.p_neg, 
                reg_scale=self.reg_scale
            )
            self.validator = UKGE_RECT_VALIDATOR()


    def train(self, epochs=20, save_every_epoch=10, lr=0.001, data_dir=""):
        sess = tf.Session()  # show device info
        sess.run(tf.global_variables_initializer())

        num_batch = self.this_data.triples.shape[0] // self.batch_size
        print('Number of batches per epoch: %d' % num_batch)

        train_losses = []  # [[every epoch, loss]]
        val_losses = []  # [[saver epoch, loss]]

        for epoch in range(1, epochs + 1):
            train_loss = self.train1epoch(sess, num_batch, lr, epoch)
            train_losses.append([epoch, train_loss])

            if np.isnan(train_loss):
                print("Nan loss. Training collapsed.")
                return

            if epoch % save_every_epoch == 0:
                # save model
                this_save_path = self.tf_parts._saver.save(sess, self.save_path, global_step=epoch)  # save model
                self.this_data.save(self.data_save_path)  # save data
                print('VALIDATE AND SAVE MODELS:')
                print("Model saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))

                # validation error
                val_loss, val_loss_neg, mean_ndcg, mean_exp_ndcg = self.get_val_loss(epoch, sess)  # loss for testing triples and negative samples
                val_losses.append([epoch, val_loss, val_loss_neg, mean_ndcg, mean_exp_ndcg])

                # save and print metrics
                self.save_loss(train_losses, self.train_loss_path, columns=['epoch', 'training_loss'])
                self.save_loss(val_losses, self.val_loss_path, columns=['val_epoch', 'mse', 'mse_neg', 'ndcg(linear)', 'ndcg(exp)'])

        this_save_path = self.tf_parts._saver.save(sess, self.save_path)
        with sess.as_default():
            ht_embeddings = self.tf_parts._ht.eval()
            r_embeddings = self.tf_parts._r.eval()
        print("Model saved in file: %s" % this_save_path)
        sess.close()
        return ht_embeddings, r_embeddings

    def get_val_loss(self, epoch, sess): # validation error
        self.validator.build_by_var(self.this_data.val_triples, self.tf_parts, self.this_data, sess=sess)

        if not hasattr(self.validator, 'hr_map'):
            self.validator.load_hr_map(param.data_dir(), 'test.tsv', ['train.tsv', 'val.tsv', 'test.tsv'])
        if not hasattr(self.validator, 'hr_map_sub'):
            hr_map200 = self.validator.get_fixed_hr(n=200)  # use smaller size for faster validation
        else:
            hr_map200 = self.validator.hr_map_sub

        mean_ndcg, mean_exp_ndcg = self.validator.mean_ndcg(hr_map200)
        mse = self.validator.get_mse(save_dir=self.save_dir, epoch=epoch, toprint=self.verbose)
        mse_neg = self.validator.get_mse_neg(self.neg_per_positive)
        return mse, mse_neg, mean_ndcg, mean_exp_ndcg

    def save_loss(self, losses, filename, columns):
        df = pd.DataFrame(losses, columns=columns)
        df.to_csv(filename, index=False)

    def train1epoch(self, sess, num_batch, lr, epoch):
        batch_time = 0
        generated_batch = self.batchloader.gen_batch(forever=True)
        train_loss = []

        for batch_id in range(num_batch):
            batch = next(generated_batch)
            A_h_index, A_r_index, A_t_index, A_w, A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch
            time00 = time.time()
            soft_h_index, soft_r_index, soft_t_index, soft_w_index = self.batchloader.gen_psl_samples()  # length: param.n_psl
            batch_time += time.time() - time00
            _, gradient, batch_loss, psl_mse, mse_pos, mse_neg, main_loss, psl_prob, psl_mse_each, rule_prior = sess.run(
                [
                    self.tf_parts._train_op, 
                    self.tf_parts._gradient,
                    self.tf_parts._A_loss, # A_loss: Main Loss + PSL Loss
                    self.tf_parts.psl_mse, 
                    self.tf_parts._f_score_h, 
                    self.tf_parts._f_score_hn,
                    self.tf_parts.main_loss, 
                    self.tf_parts.psl_prob, 
                    self.tf_parts.psl_error_each,
                    self.tf_parts.prior_psl0
                ],
                feed_dict={
                    self.tf_parts._A_h_index: A_h_index,
                    self.tf_parts._A_r_index: A_r_index,
                    self.tf_parts._A_t_index: A_t_index,
                    self.tf_parts._A_w: A_w,
                    self.tf_parts._A_neg_hn_index: A_neg_hn_index,
                    self.tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                    self.tf_parts._A_neg_t_index: A_neg_t_index,
                    self.tf_parts._A_neg_h_index: A_neg_h_index,
                    self.tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                    self.tf_parts._A_neg_tn_index: A_neg_tn_index,
                    self.tf_parts._soft_h_index: soft_h_index,
                    self.tf_parts._soft_r_index: soft_r_index,
                    self.tf_parts._soft_t_index: soft_t_index,
                    self.tf_parts._soft_w: soft_w_index, 
                    self.tf_parts._lr: lr # Learning Rate
                })
            param.prior_psl = rule_prior
            train_loss.append(batch_loss)
            if ((batch_id + 1) % 50 == 0) or batch_id == num_batch - 1:
                print('process: %d / %d. Epoch %d' % (batch_id + 1, num_batch, epoch))

        this_total_loss = np.sum(train_loss) / len(train_loss)
        print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
        return this_total_loss
