''' Module for held-out test.'''

import sys
import numpy as np
import tensorflow.compat.v1 as tf
import heapq as HP
import pandas as pd
import os
import data
import time
import pickle
import random
import sklearn
from numpy import linalg as LA
from scipy.special import expit as sigmoid
from os.path import join
from sklearn import tree
from src.model import UKGE_LOGI, UKGE_RECT
tf.disable_v2_behavior()

# This class is used to load and combine a model and a Data object, and provides some useful methods for training
class Validator(object):
    class IndexScore:
        """
        The score of a tail when h and r is given.
        It's used in the ranking task to facilitate comparison and sorting.
        Print w as 3 digit precision float.
        """

        def __init__(self, index, score):
            self.index = index
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

        def __repr__(self):
            # return "(index: %d, w:%.3f)" % (self.index, self.score)
            return "(%d, %.3f)" % (self.index, self.score)

        def __str__(self):
            return "(index: %d, w:%.3f)" % (self.index, self.score)

    def __init__(self):
        self.model = None
        self.this_data = None
        self.vec_c = np.array([0])
        self.vec_r = np.array([0])
        # below for test data
        self.test_triples = np.array([0])
        self.test_triples_group = {}

    # abstract method
    def build_by_var(self, test_data, tf_model, this_data, sess):
        raise NotImplementedError("Fatal Error: This model' validator didn't implement its build_by_var() function!")

    def load_hr_map(self, data_dir, hr_base_file, supplement_t_files, splitter='\t', line_end='\n'):
        """
        Initialize self.hr_map.
        Load self.hr_map={h:{r:t:w}}}, not restricted to test data
        :param hr_base_file: Get self.hr_map={h:r:{t:w}}} from the file.
        :param supplement_t_files: Add t(only t) to self.hr_map. Don't add h or r.
        :return:
        """
        self.hr_map = {}
        with open(join(data_dir, hr_base_file)) as f:
            for line in f:
                line = line.rstrip(line_end).split(splitter)
                h = self.this_data.con_str2index(line[0])
                r = self.this_data.rel_str2index(line[1])
                t = self.this_data.con_str2index(line[2])
                w = float(line[3])
                # construct hr_map
                if self.hr_map.get(h) == None:
                    self.hr_map[h] = {}
                if self.hr_map[h].get(r) == None:
                    self.hr_map[h][r] = {t: w}
                else:
                    self.hr_map[h][r][t] = w

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

        for file in supplement_t_files:
            with open(join(data_dir, file)) as f:
                for line in f:
                    line = line.rstrip(line_end).split(splitter)
                    h = self.this_data.con_str2index(line[0])
                    r = self.this_data.rel_str2index(line[1])
                    t = self.this_data.con_str2index(line[2])
                    w = float(line[3])

                    # update hr_map
                    if h in self.hr_map and r in self.hr_map[h]:
                        self.hr_map[h][r][t] = w

    def save_hr_map(self, outputfile):
        """
        Print to file for debugging. (not applicable for reloading)
        Prerequisite: self.hr_map has been loaded.
        :param outputfile:
        :return:
        """
        if self.hr_map is None:
            raise ValueError("Validator.hr_map hasn't been loaded! Use Validator.load_hr_map() to load it.")

        with open(outputfile, 'w') as f:
            for h in self.hr_map:
                for r in self.hr_map[h]:
                    tw_truth = self.hr_map[h][r]  # {t:w}
                    tw_list = [self.IndexScore(t, w) for t, w in tw_truth.items()]
                    tw_list.sort(reverse=True)  # descending on w
                    f.write('h: %d, r: %d\n' % (h, r))
                    f.write(str(tw_list) + '\n')

    def load_test_data(self, filename, splitter='\t', line_end='\n'):
        num_lines = 0
        triples = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 4:
                continue
            num_lines += 1
            h = self.this_data.con_str2index(line[0])
            r = self.this_data.rel_str2index(line[1])
            t = self.this_data.con_str2index(line[2])
            w = float(line[3])
            if h is None or r is None or t is None or w is None:
                continue
            triples.append([h, r, t, w])

            # add to group
            if self.test_triples_group.get(r) == None:
                self.test_triples_group[r] = [(h, r, t, w)]
            else:
                self.test_triples_group[r].append((h, r, t, w))

        # Note: test_triples will be a np.float64 array! (because of the type of w)
        # Take care of the type of hrt when unpacking.
        self.test_triples = np.array(triples)
        print("Loaded test data from %s, %d out of %d." % (filename, len(triples), num_lines))

    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]

    def con_index2str(self, str):
        return self.this_data.con_index2str(str)

    def rel_index2str(self, str):
        return self.this_data.rel_index2str(str)

    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec = self.con_index2vec(h)
        rvec = self.rel_index2vec(r)
        tvec = self.con_index2vec(t)
        return hvec, rvec, tvec

    # Abstract method. Different scoring function for different models.
    def get_score(self, h, r, t):
        raise NotImplementedError("get_score() is not defined in this model's validator")

    # Abstract method. Different scoring function for different models.
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        raise NotImplementedError("get_score_batch() is not defined in this model's validator")

<<<<<<< HEAD
    def get_mse(self, toprint=False, save_dir='', epoch=0):
=======
    def get_mse(self, verbose=True, save_dir='', epoch=0):
        "hi"
>>>>>>> timeline2
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        mse = np.sum(np.square(scores - w_batch))
        mse = mse / N

        return mse


    def get_mse_neg(self, neg_per_positive):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # negative samples
        # (score - 0)^2
        all_neg_hn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "h")
        all_neg_tn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "t")
        neg_hn_batch, neg_rel_hn_batch, \
        negt_batch, negh_batch, \
        neg_rel_tn_batch, neg_tn_batch \
            = all_neg_hn_batch[:, :, 0].astype(int), \
              all_neg_hn_batch[:, :, 1].astype(int), \
              all_neg_hn_batch[:, :, 2].astype(int), \
              all_neg_tn_batch[:, :, 0].astype(int), \
              all_neg_tn_batch[:, :, 1].astype(int), \
              all_neg_tn_batch[:, :, 2].astype(int)
        scores_hn = self.get_score_batch(neg_hn_batch, neg_rel_hn_batch, negt_batch, isneg2Dbatch=True)
        scores_tn = self.get_score_batch(negh_batch, neg_rel_tn_batch, neg_tn_batch, isneg2Dbatch=True)
        mse_hn = np.sum(np.mean(np.square(scores_hn - 0), axis=1)) / N
        mse_tn = np.sum(np.mean(np.square(scores_tn - 0), axis=1)) / N

        mse = (mse_hn + mse_tn) / 2
        return mse


    def con_index2vec_batch(self, indices):
        return np.squeeze(self.vec_c[[indices], :])

    def rel_index2vec_batch(self, indices):
        return np.squeeze(self.vec_r[[indices], :])

    def get_t_ranks(self, h, r, ts):
        """
        Given some t index, return the ranks for each t
        :return:
        """
        # prediction
        scores = np.array([self.get_score(h, r, t) for t in ts])  # predict scores for t from ground truth

        ranks = np.ones(len(ts), dtype=int)  # initialize rank as all 1

        N = self.vec_c.shape[0]  # pool of t: all concept vectors
        for i in range(N):  # compute scores for all concept vectors as t
            score_i = self.get_score(h, r, i)
            rankplus = (scores < score_i).astype(int)  # rank+1 if score<score_i
            ranks += rankplus

        return ranks

    def ndcg(self, h, r, tw_truth):
        """
        Compute nDCG(normalized discounted cummulative gain)
        sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
        :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
        :return:
        """
        # prediction
        ts = [tw.index for tw in tw_truth]
        ranks = self.get_t_ranks(h, r, ts)

        # linear gain
        gains = np.array([tw.score for tw in tw_truth])
        discounts = np.log2(ranks + 1)
        discounted_gains = gains / discounts
        dcg = np.sum(discounted_gains)  # discounted cumulative gain
        # normalize
        max_possible_dcg = np.sum(gains / np.log2(np.arange(len(gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

        # exponential gain
        exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
        exp_discounted_gains = exp_gains / discounts
        exp_dcg = np.sum(exp_discounted_gains)
        # normalize
        exp_max_possible_dcg = np.sum(
            exp_gains / np.log2(np.arange(len(exp_gains)) + 2))  # when ranks = [1, 2, ...len(truth)]
        exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

        return ndcg, exp_ndcg

    def mean_ndcg(self, hr_map):
        """
        :param hr_map: {h:{r:{t:w}}}
        :return:
        """
        ndcg_sum = 0  # nDCG with linear gain
        exp_ndcg_sum = 0  # nDCG with exponential gain
        count = 0

        t0 = time.time()

        # debug ndcg
        res = []  # [(h,r,tw_truth, ndcg)]

        for h in hr_map:
            for r in hr_map[h]:
                tw_dict = hr_map[h][r]  # {t:w}
                tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                tw_truth.sort(reverse=True)  # descending on w
                ndcg, exp_ndcg = self.ndcg(h, r, tw_truth)  # nDCG with linear gain and exponential gain
                ndcg_sum += ndcg
                exp_ndcg_sum += exp_ndcg
                count += 1
                ranks = self.get_t_ranks(h, r, [tw.index for tw in tw_truth])
                res.append((h,r,tw_truth, ndcg, ranks))

        return ndcg_sum / count, exp_ndcg_sum / count

    def get_fixed_hr(self, outputdir=None, n=500):
        hr_map500 = {}
        dict_keys = []
        for h in self.hr_map.keys():
            for r in self.hr_map[h].keys():
                dict_keys.append([h, r])

        dict_keys = sorted(dict_keys, key=lambda x: len(self.hr_map[x[0]][x[1]]), reverse=True)
        dict_final_keys = []

        for i in range(2525):
            dict_final_keys.append(dict_keys[i])

        count = 0
        for i in range(n):
            temp_key = random.choice(dict_final_keys)
            h = temp_key[0]
            r = temp_key[1]
            for t in self.hr_map[h][r]:
                w = self.hr_map[h][r][t]
                if hr_map500.get(h) == None:
                    hr_map500[h] = {}
                if hr_map500[h].get(r) == None:
                    hr_map500[h][r] = {t: w}
                else:
                    hr_map500[h][r][t] = w

        for h in hr_map500.keys():
            for r in hr_map500[h].keys():
                count = count + 1

        self.hr_map_sub = hr_map500

        if outputdir is not None:
            with open(outputdir, 'wb') as handle:
                pickle.dump(hr_map500, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return hr_map500



class UKGE_LOGI_VALIDATOR(Validator):
    def __init__(self, ):
        Validator.__init__(self)

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.model = tf_model

        value_ht, value_r, w, b = sess.run([self.model._ht, self.model._r, self.model.w, self.model.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

    # override
    def get_score(self, h, r, t):
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        return sigmoid(self.w*np.sum(np.multiply(np.multiply(hvec, tvec), rvec))+self.b)

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        return sigmoid(self.w*np.sum(np.multiply(np.multiply(hvecs, tvecs), rvecs), axis=axis)+self.b)


class UKGE_RECT_VALIDATOR(Validator):
    def __init__(self, ):
        Validator.__init__(self)

    # override
    def build_by_var(self, test_data, tf_model, this_data, sess=tf.Session()):
        """
        use data and model in memory.
        get self.vec_c (vectors for concepts), and self.vec_r(vectors for relations)
        :return:
        """
        self.this_data = this_data  # data.Data()

        self.test_triples = test_data
        self.model = tf_model

        value_ht, value_r, w, b = sess.run([self.model._ht, self.model._r, self.model.w, self.model.b])  # extract values.
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.w = w
        self.b = b

    # override
    def get_score(self, h, r, t):
        # no sigmoid
        hvec, rvec, tvec = self.vecs_from_triples(h, r, t)
        return self.w * np.sum(np.multiply(np.multiply(hvec, tvec), rvec)) + self.b

    # override
    def get_score_batch(self, h_batch, r_batch, t_batch, isneg2Dbatch=False):
        # no sigmoid
        hvecs = self.con_index2vec_batch(h_batch)
        rvecs = self.rel_index2vec_batch(r_batch)
        tvecs = self.con_index2vec_batch(t_batch)
        if isneg2Dbatch:
            axis = 2  # axis for reduce_sum
        else:
            axis = 1
        return self.w * np.sum(np.multiply(np.multiply(hvecs, tvecs), rvecs), axis=axis) + self.b


    def bound_score(self, scores):
        """
        scores<0 =>0
        score>1 => 1
        :param scores:
        :return:
        """
        return np.minimum(np.maximum(scores, 0), 1)


    def get_mse(self, toprint=False, save_dir='', epoch=0):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # existing triples
        # (score - w)^2
        h_batch = test_triples[:, 0].astype(int)
        r_batch = test_triples[:, 1].astype(int)
        t_batch = test_triples[:, 2].astype(int)
        w_batch = test_triples[:, 3]
        scores = self.get_score_batch(h_batch, r_batch, t_batch)
        scores = self.bound_score(scores)
        mse = np.sum(np.square(scores - w_batch))

        mse = mse / N

        return mse


    def get_mse_neg(self, neg_per_positive):
        test_triples = self.test_triples
        N = test_triples.shape[0]

        # negative samples
        # (score - 0)^2
        all_neg_hn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "h")
        all_neg_tn_batch = self.this_data.corrupt_batch(test_triples, neg_per_positive, "t")
        neg_hn_batch, neg_rel_hn_batch, \
        negt_batch, negh_batch, \
        neg_rel_tn_batch, neg_tn_batch \
            = all_neg_hn_batch[:, :, 0].astype(int), \
              all_neg_hn_batch[:, :, 1].astype(int), \
              all_neg_hn_batch[:, :, 2].astype(int), \
              all_neg_tn_batch[:, :, 0].astype(int), \
              all_neg_tn_batch[:, :, 1].astype(int), \
              all_neg_tn_batch[:, :, 2].astype(int)
        scores_hn = self.get_score_batch(neg_hn_batch, neg_rel_hn_batch, negt_batch, isneg2Dbatch=True)
        scores_tn = self.get_score_batch(negh_batch, neg_rel_tn_batch, neg_tn_batch, isneg2Dbatch=True)

        scores_hn = self.bound_score(scores_hn)
        scores_tn = self.bound_score(scores_tn)

        mse_hn = np.sum(np.mean(np.square(scores_hn - 0), axis=1)) / N
        mse_tn = np.sum(np.mean(np.square(scores_tn - 0), axis=1)) / N

        mse = (mse_hn + mse_tn) / 2
        return mse

