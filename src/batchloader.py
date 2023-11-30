"""Processing of data."""

import numpy as np
import pickle
import pandas as pd
from os.path import join
from src import param

class BatchLoader():
    def __init__(self, data_obj, batch_size, neg_per_positive):
        self.this_data = data_obj  # Data() object
        self.shuffle = True
        self.batch_size = batch_size
        self.neg_per_positive = neg_per_positive
        self.n_soft_samples = param.n_psl  # number of samples per batch

    def gen_psl_samples(self):
        # samples from probabilistic soft logic
        softlogics = self.this_data.soft_logic_triples  # [[A, B, base]]
        triple_indices = np.random.randint(0, softlogics.shape[0], size=self.n_soft_samples)
        samples = softlogics[triple_indices,:]
        soft_h, soft_r, soft_t, soft_lb = samples[:,0].astype(int),samples[:,1].astype(int),samples[:,2].astype(int), samples[:,3]
        soft_sample_batch = (soft_h, soft_r, soft_t, soft_lb)
        return soft_sample_batch

    def gen_batch(self, forever=False, shuffle=True, negsampler=None):
        """
        :param ht_embedding: for kNN negative sampling
        :return:
        """
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples  # np.float64 [[h,r,t,w]]
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):
                batch = triples[i: i + self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]),
                                           axis=0)
                    assert batch.shape[0] == self.batch_size

                h_batch, r_batch, t_batch, w_batch = batch[:, 0].astype(int), batch[:, 1].astype(int), batch[:,
                                                                                                       2].astype(
                    int), batch[:, 3]
                hrt_batch = batch[:, 0:3].astype(int)

                if negsampler is None:
                    neg_hn_batch, neg_rel_hn_batch, \
                    neg_t_batch, neg_h_batch, \
                    neg_rel_tn_batch, neg_tn_batch \
                        = self.corrupt_batch(h_batch, r_batch, t_batch)
                else:
                    # neg_per_positive controlled by sampler
                    all_neg_hn_batch = negsampler.knn_negative_batch(hrt_batch, "h")
                    all_neg_tn_batch = negsampler.knn_negative_batch(hrt_batch, "t")

                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(
                    np.int64), w_batch.astype(
                    np.float32), \
                      neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), \
                      neg_t_batch.astype(np.int64), neg_h_batch.astype(np.int64), \
                      neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)
            if not forever:
                break

    def corrupt_batch(self, h_batch, r_batch, t_batch):
        N = self.this_data.num_cons()  # number of entities

        neg_hn_batch = np.random.randint(0, N, size=(
        self.batch_size, self.neg_per_positive))  # random index without filtering
        neg_rel_hn_batch = np.tile(r_batch, (self.neg_per_positive, 1)).transpose()  # copy
        neg_t_batch = np.tile(t_batch, (self.neg_per_positive, 1)).transpose()

        neg_h_batch = np.tile(h_batch, (self.neg_per_positive, 1)).transpose()
        neg_rel_tn_batch = neg_rel_hn_batch
        neg_tn_batch = np.random.randint(0, N, size=(self.batch_size, self.neg_per_positive))

        return neg_hn_batch, neg_rel_hn_batch, neg_t_batch, neg_h_batch, neg_rel_tn_batch, neg_tn_batch
