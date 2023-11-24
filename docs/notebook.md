# UKGE Notebook

This Notebook is extracted with only the crucial parts from UKGE Codebase so some parts like Validator, RECT, @property and other datasets like PPI5k, NL27K was not included for a thorough understanding.

Prerequisite:
- Create a new blank Notebook in Kaggle: https://kaggle.com
- Import the dataset from: https://www.kaggle.com/datasets/thala321/cn15k-dataset

## Step 1: Import libraries

Even though we install Tensorflow 2.0 as our library, but the legacy code of UKGE is implemented in Tensorflow 1.0 so a quick hack to change tensorflow version to 1.0 is done by setting `tf` as `tensorflow.compat.v1` 

```python
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
tf.disable_v2_behavior()
```

## Step 2: Set variables
```python
neg_per_positive = 10
batch_size = 1024
epochs = 20
lr=0.001
```

## Step 3: Define Data and BatchLoader

We define a new class to load the triplets in the CN15K dataset. It will have a batch function to load the data by batches instead of separate data points. The class also have the corrupt function to corrupt some samples for testing the model.

```python
class Data(object):
    '''Interfaces for holding all data.
    '''
    
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
        # recorded for model
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
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_size
                h_batch, r_batch, t_batch, w_batch = batch[:, 0].astype(int), batch[:, 1].astype(int), batch[:, 2].astype(int), batch[:, 3]
                hrt_batch = batch[:, 0:3].astype(int)
                if negsampler is None:
                    neg_hn_batch, neg_rel_hn_batch, neg_t_batch, neg_h_batch, neg_rel_tn_batch, neg_tn_batch = self.corrupt_batch(h_batch, r_batch, t_batch)
                else:
                    # neg_per_positive controlled by sampler
                    all_neg_hn_batch = negsampler.knn_negative_batch(hrt_batch, "h")
                    all_neg_tn_batch = negsampler.knn_negative_batch(hrt_batch, "t")
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), w_batch.astype(np.float32), neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), neg_t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)
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
```

## Step 4: Load data
```python
train_data = pd.read_csv('/kaggle/input/cn15k-dataset/train.tsv', sep='\t', header=None, names=['v1','relation','v2','w'])
this_data = Data()
this_data.load_data(file_train='/kaggle/input/cn15k-dataset/train.tsv', 
                file_val='/kaggle/input/cn15k-dataset/val.tsv', 
                file_psl='/kaggle/input/cn15k-dataset/softlogic.tsv')
batchloader = BatchLoader(this_data, batch_size, neg_per_positive)
```

## Step 5: Define UKGE LOGI Model

`__init__` method initializes various parameters required for the UKGE Logistic Regression model, like the number of relations `num_rels`, number of ontologies `num_cons`, embedding dimensions `dim`, batch size `batch_size`, etc. It also sets default values for certain coefficients and variables used in the model.

Then defines the placeholders for input data: `_A_*` placeholders for indices of entities and relations, `_soft_*` placeholders for uncertain graph and PSL-related data and initializes trainable variables with `ht` for entity embeddings, and `r` for relation embeddings. Embeddings for positive and negative samples are looked up from the embedding matrices using `tf.nn.embedding_lookup`. With `main_loss` computes the main loss for the model. It uses the embeddings to calculate scores for positive and negative samples and computes a loss based on the difference between these scores and the expected scores `_A_w` placeholder. While `psl_loss` computes the loss related to Probabilistic Soft Logic (PSL). It involves calculating a probability based on uncertain graph embeddings and then calculating the error against expected soft constraints. And the `_opt` uses the Adam optimizer to minimize the combined loss `_A_loss`, which includes both the main loss and the PSL loss. `_train_op` applies the gradients computed by the optimizer to update the model parameters during training.

```python
class UKGE_LOGI(object):
    '''
    UKGE Logistic Regression
    
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
            self._ht = ht = tf.get_variable(name='ht', shape=[self._num_cons, self._dim], dtype=tf.float32)
            self._r = r = tf.get_variable(name='r', shape=[self._num_rels, self._dim], dtype=tf.float32)
            self._A_h_index = tf.placeholder(dtype=tf.int64, shape=[self._batch_size], name='A_h_index')
            self._A_r_index = tf.placeholder(dtype=tf.int64, shape=[self._batch_size], name='A_r_index')
            self._A_t_index = tf.placeholder(dtype=tf.int64, shape=[self._batch_size], name='A_t_index')
            # for uncertain graph
            self._A_w = tf.placeholder(dtype=tf.float32, shape=[self._batch_size], name='_A_w')
            self._A_neg_hn_index = tf.placeholder(dtype=tf.int64, shape=(self._batch_size, self._neg_per_positive), name='A_neg_hn_index')
            self._A_neg_rel_hn_index = tf.placeholder(dtype=tf.int64, shape=(self._batch_size, self._neg_per_positive), name='A_neg_rel_hn_index')
            self._A_neg_t_index = tf.placeholder(dtype=tf.int64, shape=(self._batch_size, self._neg_per_positive), name='A_neg_t_index')
            self._A_neg_h_index = tf.placeholder(dtype=tf.int64, shape=(self._batch_size, self._neg_per_positive), name='A_neg_h_index')
            self._A_neg_rel_tn_index = tf.placeholder(dtype=tf.int64, shape=(self._batch_size, self._neg_per_positive), name='A_neg_rel_tn_index')
            self._A_neg_tn_index = tf.placeholder(dtype=tf.int64, shape=(self._batch_size, self._neg_per_positive), name='A_neg_tn_index')
            # no normalization
            self._h_batch = tf.nn.embedding_lookup(ht, self._A_h_index)
            self._t_batch = tf.nn.embedding_lookup(ht, self._A_t_index)
            self._r_batch = tf.nn.embedding_lookup(r, self._A_r_index)
            self._neg_hn_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_hn_index)
            self._neg_rel_hn_batch = tf.nn.embedding_lookup(r, self._A_neg_rel_hn_index)
            self._neg_t_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_t_index)
            self._neg_h_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_h_index)
            self._neg_rel_tn_batch = tf.nn.embedding_lookup(r, self._A_neg_rel_tn_index)
            self._neg_tn_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_tn_index)
            # psl batches
            self._soft_h_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_h_index')
            self._soft_r_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_r_index')
            self._soft_t_index = tf.placeholder(dtype=tf.int64, shape=[self._soft_size], name='soft_t_index')
            # for uncertain graph and psl
            self._soft_w = tf.placeholder(dtype=tf.float32, shape=[self._soft_size], name='soft_w_lower_bound')
            self._soft_h_batch = tf.nn.embedding_lookup(ht, self._soft_h_index)
            self._soft_t_batch = tf.nn.embedding_lookup(ht, self._soft_t_index)
            self._soft_r_batch = tf.nn.embedding_lookup(r, self._soft_r_index)

        # Compute Main Loss
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        self._htr = tf.reduce_sum(tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),"r_product"), 1)
        self._f_prob_h = tf.sigmoid(self.w * self._htr + self.b) # Logistic regression
        self._f_score_h = tf.square(tf.subtract(self._f_prob_h, self._A_w))
        self._f_prob_hn = tf.sigmoid(self.w * (tf.reduce_sum( tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)) + self.b)
        self._f_score_hn = tf.reduce_mean(tf.square(self._f_prob_hn), 1)
        self._f_prob_tn = tf.sigmoid(self.w * (tf.reduce_sum(tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)) + self.b)
        self._f_score_tn = tf.reduce_mean(tf.square(self._f_prob_tn), 1)
        self.main_loss = (tf.reduce_sum(tf.add(tf.divide(tf.add(self._f_score_tn, self._f_score_hn), 2) * self._p_neg, self._f_score_h))) / self._batch_size
        
        # Compute PSL Loss
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(tf.multiply(self._soft_r_batch, tf.multiply(self._soft_h_batch, self._soft_t_batch)), 1)+self.b)
        self.prior_psl0 = tf.constant(self._prior_psl, tf.float32)
        self.psl_error_each = tf.square(tf.maximum(self._soft_w + self.prior_psl0 - self.psl_prob, 0))
        self.psl_mse = tf.reduce_mean(self.psl_error_each)
        self.psl_loss = self.psl_mse * self._p_psl

        # Optimizer
        self._A_loss = tf.add(self.main_loss, self.psl_loss)
        self._lr = tf.placeholder(tf.float32)
        self._opt = tf.train.AdamOptimizer(self._lr)
        self._gradient = self._opt.compute_gradients(self._A_loss) 
        self._train_op = self._opt.apply_gradients(self._gradient)
```

## Step 6: Model
```python
model = UKGE_LOGI(num_rels=this_data.num_rels(),
                num_cons=this_data.num_cons(),
                dim=this_data.dim,
                batch_size=this_data.batch_size, 
                neg_per_positive=10, 
                p_neg=1)
model.build()
```

## Step 7: Training

A session is created and started using `tf.Session()` and `Session.run` takes the operations we created and data to be fed as parameters and it returns the result. Only after running `tf.global_variables_initializer()` in a session will the variables hold the values you told them to hold.

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
num_batch = this_data.triples.shape[0] // batch_size
print('Number of batches per epoch: %d' % num_batch)
train_losses = []  # [[every epoch, loss]]
val_losses = []  # [[saver epoch, loss]]
for epoch in range(1, epochs + 1):
    epoch_batches = batchloader.gen_batch(forever=True)
    epoch_loss = []
    for batch_id in range(num_batch):
        batch = next(epoch_batches)
        A_h_index, A_r_index, A_t_index, A_w, A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch
        soft_h_index, soft_r_index, soft_t_index, soft_w_index = batchloader.gen_psl_samples()  # length: param.n_psl
        _, gradient, batch_loss, psl_mse, mse_pos, mse_neg, main_loss, psl_prob, psl_mse_each, rule_prior = sess.run(
            [
                model._train_op, 
                model._gradient,
                model._A_loss, # A_loss: Main Loss + PSL Loss
                model.psl_mse, 
                model._f_score_h, 
                model._f_score_hn,
                model.main_loss, 
                model.psl_prob, 
                model.psl_error_each,
                model.prior_psl0
            ],
            feed_dict={
                model._A_h_index: A_h_index,
                model._A_r_index: A_r_index,
                model._A_t_index: A_t_index,
                model._A_w: A_w,
                model._A_neg_hn_index: A_neg_hn_index,
                model._A_neg_rel_hn_index: A_neg_rel_hn_index,
                model._A_neg_t_index: A_neg_t_index,
                model._A_neg_h_index: A_neg_h_index,
                model._A_neg_rel_tn_index: A_neg_rel_tn_index,
                model._A_neg_tn_index: A_neg_tn_index,
                model._soft_h_index: soft_h_index,
                model._soft_r_index: soft_r_index,
                model._soft_t_index: soft_t_index,
                model._soft_w: soft_w_index, 
                model._lr: lr # Learning Rate
            })
        prior_psl = rule_prior
        epoch_loss.append(batch_loss)
        if ((batch_id + 1) % 50 == 0) or batch_id == num_batch - 1:
            print('process: %d / %d. Epoch %d' % (batch_id + 1, num_batch, epoch))
    this_total_loss = np.sum(epoch_loss) / len(epoch_loss)
    print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
```

Here's the result:

![image](https://github.com/hughiephan/UKGE/assets/16631121/f0b4d7f5-62c4-4755-b9a5-85f1e54fef43)
