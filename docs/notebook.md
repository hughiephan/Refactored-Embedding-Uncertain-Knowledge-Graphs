# UKGE Notebook

Dataset is from: https://www.kaggle.com/datasets/thala321/cn15k-dataset

## Define model
```python
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import pickle
from scipy.special import expit as sigmoid
from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
tf.disable_v2_behavior()

class Data(object):
    '''The abstract class that defines interfaces for holding all data.
    '''

    def __init__(self):
        self.cons = [] # Concept vocab
        self.rels = [] # Relation vocab
        # Transitive rels vocab
        self.index_cons = {}  # {string: index}
        self.index_rels = {}  # {string: index}
        # save triples as array of indices
        self.triples = np.array([0])  # training dataset
        self.val_triples = np.array([0])  # validation dataset
        self.soft_logic_triples = np.array([0])

        # (h,r,t) tuples(int), no w
        # set containing train, val, test (for negative sampling).
        self.triples_record = set([])
        self.weights = np.array([0])

        self.neg_triples = np.array([0])
        # map for sigma
        # head per tail and tail per head (for each relation). used for bernoulli negative sampling
        self.hpt = np.array([0])
        self.tph = np.array([0])
        # recorded for tf_parts
        self.dim = 64
        self.batch_size = 1024
        self.L1 = False

    def load_triples(self, filename, splitter='\t', line_end='\n'):
        '''Load the dataset'''
        triples = []
        last_c = -1
        last_r = -1
        hr_map = {}
        tr_map = {}

        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if self.index_cons.get(line[0]) == None:
                self.cons.append(line[0])
                last_c += 1
                self.index_cons[line[0]] = last_c
            if self.index_cons.get(line[2]) == None:
                self.cons.append(line[2])
                last_c += 1
                self.index_cons[line[2]] = last_c
            if self.index_rels.get(line[1]) == None:
                self.rels.append(line[1])
                last_r += 1
                self.index_rels[line[1]] = last_r
            h = self.index_cons[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_cons[line[2]]
            w = float(line[3])

            triples.append([h, r, t, w])
            self.triples_record.add((h, r, t))
        return np.array(triples)

    def load_data(self, file_train, file_val, file_psl=None, splitter='\t', line_end='\n'):

        self.triples = self.load_triples(file_train, splitter, line_end)
        self.val_triples = self.load_triples(file_val, splitter, line_end)

        if file_psl is not None:
            self.soft_logic_triples = self.load_triples(file_psl, splitter, line_end)

        # calculate tph and hpt
        tph_array = np.zeros((len(self.rels), len(self.cons)))
        hpt_array = np.zeros((len(self.rels), len(self.cons)))
        for h_, r_, t_, w in self.triples:  # only training data
            h, r, t = int(h_), int(r_), int(t_)
            tph_array[r][h] += 1.
            hpt_array[r][t] += 1.
        self.tph = np.mean(tph_array, axis=1)
        self.hpt = np.mean(hpt_array, axis=1)
        # print("-- total number of entities:", len(self.cons))

    def num_cons(self):
        '''Returns number of ontologies.

        This means all ontologies have index that 0 <= index < num_onto().
        '''
        return len(self.cons)

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return len(self.rels)


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
            self._A_h_index = A_h_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_t_index')
            # for uncertain graph
            self._A_w = tf.placeholder(dtype=tf.float32, shape=[self.batch_size], name='_A_w')
            self._A_neg_hn_index = A_neg_hn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_hn_index')
            self._A_neg_rel_hn_index = A_neg_rel_hn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_rel_hn_index')
            self._A_neg_t_index = A_neg_t_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_t_index')
            self._A_neg_h_index = A_neg_h_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_h_index')
            self._A_neg_rel_tn_index = A_neg_rel_tn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_rel_tn_index')
            self._A_neg_tn_index = A_neg_tn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self._neg_per_positive), name='A_neg_tn_index')
            # no normalization
            self._h_batch = tf.nn.embedding_lookup(ht, A_h_index)
            self._t_batch = tf.nn.embedding_lookup(ht, A_t_index)
            self._r_batch = tf.nn.embedding_lookup(r, A_r_index)
            self._neg_hn_con_batch = tf.nn.embedding_lookup(ht, A_neg_hn_index)
            self._neg_rel_hn_batch = tf.nn.embedding_lookup(r, A_neg_rel_hn_index)
            self._neg_t_con_batch = tf.nn.embedding_lookup(ht, A_neg_t_index)
            self._neg_h_con_batch = tf.nn.embedding_lookup(ht, A_neg_h_index)
            self._neg_rel_tn_batch = tf.nn.embedding_lookup(r, A_neg_rel_tn_index)
            self._neg_tn_con_batch = tf.nn.embedding_lookup(ht, A_neg_tn_index)
            # psl batches
            self._soft_h_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_h_index')
            self._soft_r_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_r_index')
            self._soft_t_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_t_index')
            # for uncertain graph and psl
            self._soft_w = tf.placeholder(dtype=tf.float32, shape=[self._soft_size], name='soft_w_lower_bound')
            self._soft_h_batch = tf.nn.embedding_lookup(ht, self._soft_h_index)
            self._soft_t_batch = tf.nn.embedding_lookup(ht, self._soft_t_index)
            self._soft_r_batch = tf.nn.embedding_lookup(r, self._soft_r_index)

        self.define_main_loss()  # Abstract method to be overriden
        self.define_psl_loss()  # Abstract method to be overriden

        # Optimizer
        self._A_loss = tf.add(self.main_loss, self.psl_loss)
        self._lr = lr = tf.placeholder(tf.float32)
        self._opt = opt = tf.train.AdamOptimizer(lr)
        self._gradient = gradient = opt.compute_gradients(self._A_loss) 
        self._train_op = opt.apply_gradients(gradient)
        self._saver = tf.train.Saver(max_to_keep=2)

    def compute_psl_loss(self): # Will be trained in Trainer through TF Parts
        self.prior_psl0 = tf.constant(self._prior_psl, tf.float32)
        self.psl_error_each = tf.square(tf.maximum(self._soft_w + self.prior_psl0 - self.psl_prob, 0))
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

        self._htr = htr = tf.reduce_sum(tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),"r_product"), 1)

        self._f_prob_h = f_prob_h = tf.sigmoid(self.w * htr + self.b) # Logistic regression
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = tf.sigmoid(self.w * (tf.reduce_sum( tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = tf.sigmoid(self.w * (tf.reduce_sum(tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)) + self.b)
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.main_loss = (tf.reduce_sum(tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    # Override abstract method
    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(tf.multiply(self._soft_r_batch, tf.multiply(self._soft_h_batch, self._soft_t_batch)), 1)+self.b)
        self.compute_psl_loss()
```

## Load model, extract values and start training
```python
train_data = pd.read_csv('/kaggle/input/cn15k-dataset/train.tsv', sep='\t', header=None, names=['v1','relation','v2','w'])
this_data = Data()
this_data.load_data(file_train='/kaggle/input/cn15k-dataset/train.tsv', 
                    file_val='/kaggle/input/cn15k-dataset/val.tsv', 
                    file_psl='/kaggle/input/cn15k-dataset/softlogic.tsv')
        
model = UKGE_LOGI(num_rels=this_data.num_rels(),
                    num_cons=this_data.num_cons(),
                    dim=this_data.dim,
                    batch_size=this_data.batch_size, 
                    neg_per_positive=10, 
                    p_neg=1)

# Extracting Values
sess = tf.Session()
sess.run(tf.global_variables_initializer())
value_ht, value_r, w, b = sess.run([model._ht, model._r, model.w, model.b])
sess.close()
vec_c = np.array(value_ht)
vec_r = np.array(value_r)

# Training
def get_score_batch(h_batch, r_batch, t_batch, isneg2Dbatch=False):
    hvecs = np.squeeze(vec_c[[h_batch], :])
    rvecs = np.squeeze(vec_r[[r_batch], :])
    tvecs = np.squeeze(vec_c[[h_batch], :])
    if isneg2Dbatch:
        axis = 2  # axis for reduce_sum
    else:
        axis = 1
    return sigmoid(w*np.sum(np.multiply(np.multiply(hvecs, tvecs), rvecs), axis=axis)+b)

train_h, train_r, train_t = train_data['v1'].values.astype(int), train_data['relation'].values.astype(int), train_data['v2'].values.astype(int)
train_X = get_score_batch(train_h, train_r, train_t)[:, np.newaxis]  # Feature(2D, n*1)
# train_Y = train_data['w']>confT  # label (high confidence/not)
train_Y = train_data['w'] # Simplify this code because I could not find what is confT
clf = tree.DecisionTreeClassifier()
lab = preprocessing.LabelEncoder()
train_X_transformed = lab_enc.fit_transform(train_X)
train_Y_transformed = lab.fit_transform(train_Y)
clf.fit([train_X_transformed], [train_Y_transformed])
```

## Try with Validator
```python
import time

class BatchLoader():
    def __init__(self, data_obj, batch_size, neg_per_positive):
        self.this_data = data_obj  # Data() object
        self.shuffle = True
        self.batch_size = batch_size
        self.neg_per_positive = neg_per_positive
        self.n_soft_samples = 1  # number of samples per batch, n_psl

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
                    # all_neg_hn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "h")
                    # all_neg_tn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "t")

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

    
neg_per_positive = 10
batchloader = BatchLoader(this_data, batch_size, neg_per_positive)



def train1epoch(sess, num_batch, lr, epoch):
    batch_time = 0
    epoch_batches = batchloader.gen_batch(forever=True)
    epoch_loss = []

    for batch_id in range(num_batch):
        batch = next(epoch_batches)
        A_h_index, A_r_index, A_t_index, A_w, A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch
        time00 = time.time()
        soft_h_index, soft_r_index, soft_t_index, soft_w_index = batchloader.gen_psl_samples()  # length: param.n_psl
        batch_time += time.time() - time00

        # mse_pos: MSE on Positive samples
        # mse_neg: MSE on Negative samples
        _, gradient, batch_loss, psl_mse, mse_pos, mse_neg, main_loss, psl_prob, psl_mse_each, rule_prior = sess.run(
            [
                tf_parts._train_op, 
                tf_parts._gradient,
                tf_parts._A_loss, # A_loss: Main Loss + PSL Loss
                tf_parts.psl_mse, 
                tf_parts._f_score_h, 
                tf_parts._f_score_hn,
                tf_parts.main_loss, 
                tf_parts.psl_prob, 
                tf_parts.psl_error_each,
                tf_parts.prior_psl0
            ],
            feed_dict={
                tf_parts._A_h_index: A_h_index,
                tf_parts._A_r_index: A_r_index,
                tf_parts._A_t_index: A_t_index,
                tf_parts._A_w: A_w,
                tf_parts._A_neg_hn_index: A_neg_hn_index,
                tf_parts._A_neg_rel_hn_index: A_neg_rel_hn_index,
                tf_parts._A_neg_t_index: A_neg_t_index,
                tf_parts._A_neg_h_index: A_neg_h_index,
                tf_parts._A_neg_rel_tn_index: A_neg_rel_tn_index,
                tf_parts._A_neg_tn_index: A_neg_tn_index,
                tf_parts._soft_h_index: soft_h_index,
                tf_parts._soft_r_index: soft_r_index,
                tf_parts._soft_t_index: soft_t_index,
                tf_parts._soft_w: soft_w_index, 
                tf_parts._lr: lr # Learning Rate
            })
        param.prior_psl = rule_prior
        epoch_loss.append(batch_loss)
        if ((batch_id + 1) % 50 == 0) or batch_id == num_batch - 1:
            print('process: %d / %d. Epoch %d' % (batch_id + 1, num_batch, epoch))

    this_total_loss = np.sum(epoch_loss) / len(epoch_loss)
    print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
    # print('MSE on positive instances: %f, MSE on negative samples: %f' % (np.mean(mse_pos), np.mean(mse_neg)))
    return this_total_loss



batch_size = 128
epochs = 20
lr=0.001

sess = tf.Session()  # show device info
sess.run(tf.global_variables_initializer())

num_batch = this_data.triples.shape[0] // batch_size
print('Number of batches per epoch: %d' % num_batch)

train_losses = []  # [[every epoch, loss]]
val_losses = []  # [[saver epoch, loss]]

for epoch in range(1, epochs + 1):
    epoch_loss = train1epoch(sess, num_batch, lr, epoch)
    train_losses.append([epoch, epoch_loss])

    if np.isnan(epoch_loss):
        print("Nan loss. Training collapsed.")

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
```
