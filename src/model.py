"""
Tensorflow related part
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class UKGE(object):
    '''
    TensorFlow-related things.
    Keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg):
        '''
        Initialize the variables
        '''
        self._num_rels = num_rels # Number of relations
        self._num_cons = num_cons # Number of ontologies
        self._dim = dim  # Dimension of both relation and ontology.
        self._neg_per_positive = neg_per_positive # Number of negative samples per (h,r,t)
        self._p_psl = 0.2 # Coefficient
        self._p_neg = 1
        self._batch_size = batch_size
        self._epoch_loss = 0 
        self._soft_size = 1
        self._prior_psl = 0

    def build(self):
        tf.reset_default_graph()
        with tf.variable_scope("graph", initializer=tf.truncated_normal_initializer(0, 0.3)):
            # Variables (matrix of embeddings/transformations)
            self._ht = ht = tf.get_variable(name='ht', shape=[self.num_cons, self.dim], dtype=tf.float32)
            self._r = r = tf.get_variable(name='r', shape=[self.num_rels, self.dim], dtype=tf.float32)
            self.A_h_index = A_h_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_h_index')
            self.A_r_index = A_r_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_r_index')
            self.A_t_index = A_t_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_t_index')
            # for uncertain graph
            self.A_w = tf.placeholder(dtype=tf.float32, shape=[self.batch_size], name='A_w')
            self.A_neg_hn_index = A_neg_hn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_hn_index')
            self.A_neg_rel_hn_index = A_neg_rel_hn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_rel_hn_index')
            self.A_neg_t_index = A_neg_t_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_t_index')
            self.A_neg_h_index = A_neg_h_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_h_index')
            self.A_neg_rel_tn_index = A_neg_rel_tn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_rel_tn_index')
            self.A_neg_tn_index = A_neg_tn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_tn_index')
            # no normalization
            self.h_batch = tf.nn.embedding_lookup(ht, A_h_index)
            self.t_batch = tf.nn.embedding_lookup(ht, A_t_index)
            self.r_batch = tf.nn.embedding_lookup(r, A_r_index)
            self.neg_hn_con_batch = tf.nn.embedding_lookup(ht, A_neg_hn_index)
            self.neg_rel_hn_batch = tf.nn.embedding_lookup(r, A_neg_rel_hn_index)
            self.neg_t_con_batch = tf.nn.embedding_lookup(ht, A_neg_t_index)
            self.neg_h_con_batch = tf.nn.embedding_lookup(ht, A_neg_h_index)
            self.neg_rel_tn_batch = tf.nn.embedding_lookup(r, A_neg_rel_tn_index)
            self.neg_tn_con_batch = tf.nn.embedding_lookup(ht, A_neg_tn_index)
            # psl batches
            self.soft_h_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_h_index')
            self.soft_r_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_r_index')
            self.soft_t_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_t_index')
            # for uncertain graph and psl
            self.soft_w = tf.placeholder(dtype=tf.float32, shape=[self._soft_size], name='soft_w_lower_bound')
            self.softh_batch = tf.nn.embedding_lookup(ht, self.soft_h_index)
            self.softt_batch = tf.nn.embedding_lookup(ht, self.soft_t_index)
            self.softr_batch = tf.nn.embedding_lookup(r, self.soft_r_index)

        self.define_main_loss()  # Abstract method to be overriden
        self.define_psl_loss()  # Abstract method to be overriden

        # Optimizer
        self.A_loss = tf.add(self.main_loss, self.psl_loss)
        self.lr = lr = tf.placeholder(tf.float32)
        self.opt = opt = tf.train.AdamOptimizer(lr)
        self.gradient = gradient = opt.compute_gradients(self.A_loss) 
        self.train_op = opt.apply_gradients(gradient)
        self.saver = tf.train.Saver(max_to_keep=2)

    def compute_psl_loss(self): # Will be trained in Trainer through TF Parts
        self.prior_psl0 = tf.constant(self._prior_psl, tf.float32)
        self.psl_error_each = tf.square(tf.maximum(self.soft_w + self.prior_psl0 - self.psl_prob, 0))
        self.psl_mse = tf.reduce_mean(self.psl_error_each)
        self.psl_loss = self.psl_mse * self._p_psl

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size


class UKGE_LOGI(UKGE):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg):
        UKGE.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg)
        self.build()

    # Override abstract method
    def define_main_loss(self):
        # distmult on uncertain graph
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self.htr = htr = tf.reduce_sum(tf.multiply(self.r_batch, tf.multiply(self.h_batch, self.t_batch, "element_wise_multiply"),"r_product"), 1)

        self.f_prob_h = f_prob_h = tf.sigmoid(self.w * htr + self.b) # Logistic regression
        self.f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self.A_w))

        self.f_prob_hn = f_prob_hn = tf.sigmoid(self.w * (tf.reduce_sum( tf.multiply(self.neg_rel_hn_batch, tf.multiply(self.neg_hn_con_batch, self.neg_t_con_batch)), 2)) + self.b)
        self.f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self.f_prob_tn = f_prob_tn = tf.sigmoid(self.w * (tf.reduce_sum(tf.multiply(self.neg_rel_tn_batch, tf.multiply(self.neg_h_con_batch, self.neg_tn_con_batch)), 2)) + self.b)
        self.f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.main_loss = (tf.reduce_sum(tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    # Override abstract method
    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(tf.multiply(self.softr_batch, tf.multiply(self.softh_batch, self.softt_batch)), 1)+self.b)
        self.compute_psl_loss()


class UKGE_RECT(UKGE):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg):
        UKGE.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg)
        self.reg_scale = reg_scale
        self.build()

    # Override abstract method
    def define_main_loss(self): 
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self.htr = htr = tf.reduce_sum(tf.multiply(self.r_batch, tf.multiply(self.h_batch, self.t_batch, "element_wise_multiply"),"r_product"), 1)

        self.f_prob_h = f_prob_h = self.w * htr + self.b
        self.f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self.A_w))

        self.f_prob_hn = f_prob_hn = self.w * (tf.reduce_sum(tf.multiply(self.neg_rel_hn_batch, tf.multiply(self.neg_hn_con_batch, self.neg_t_con_batch)), 2)) + self.b
        self.f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self.f_prob_tn = f_prob_tn = self.w * (tf.reduce_sum(tf.multiply(self.neg_rel_tn_batch, tf.multiply(self.neg_h_con_batch, self.neg_tn_con_batch)), 2)) + self.b
        self.f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.this_loss = this_loss = (tf.reduce_sum(tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size
        self.regularizer = regularizer = tf.add(tf.add(tf.divide(tf.nn.l2_loss(self.h_batch), self.batch_size), tf.divide(tf.nn.l2_loss(self.t_batch), self.batch_size)), tf.divide(tf.nn.l2_loss(self.r_batch), self.batch_size)) # L2 regularizer
        self.main_loss = tf.add(this_loss, self.reg_scale * regularizer)

    # Override abstract method
    def define_psl_loss(self): 
        self.psl_prob = self.w * tf.reduce_sum(tf.multiply(self.softr_batch, tf.multiply(self.softh_batch, self.softt_batch)), 1) + self.b
        self.compute_psl_loss()
