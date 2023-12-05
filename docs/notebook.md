# UKGE Notebook

![image](https://github.com/hughiephan/UKGE/assets/16631121/8f6e9632-3342-461c-b9cb-4611d8a29c88)

This Notebook is extracted with only the crucial parts from UKGE Codebase so some parts like Validator, RECT, @property and other datasets like PPI5k, NL27K was not included for a thorough understanding. If you want to run on Validation Data as well, please run the UKGE Codebase as this Notebook is focus on explaining the concept not for benchmarking.

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
dim = 64
batch_size = 1024
cons = []  # Concept dictionary or vocabulary
rels = []  # Relation dictionary or vocabulary
index_cons = {}  # {Concept: index of concept}
index_rels = {}  # {Relation: index of relation}
triples = np.array([0])  # Training dataset
soft_logic_triples = np.array([0])  # Soft logic dataset
```

## Step 3: Define Data

We define a new class to load the triplets in the CN15K dataset. It will have a batch function to load the data by batches instead of separate data points. The class also have the corrupt function to corrupt some samples for testing the model.

![Untitled-2023-07-31-0913](https://github.com/hughiephan/UKGE/assets/16631121/e72b5d6e-5c31-4722-8bce-e42023dd7dd3)

```python
def load_triples(filename):
    triples = []
    last_c = -1
    last_r = -1
    for line in open(filename):
        line = line.rstrip('\n').split('\t')
        if index_cons.get(line[0]) is None:
            cons.append(line[0])
            last_c += 1
            index_cons[line[0]] = last_c
        if index_cons.get(line[2]) is None:
            cons.append(line[2])
            last_c += 1
            index_cons[line[2]] = last_c
        if index_rels.get(line[1]) is None:
            rels.append(line[1])
            last_r += 1
            index_rels[line[1]] = last_r
        h = index_cons[line[0]]
        r = index_rels[line[1]]
        t = index_cons[line[2]]
        w = float(line[3])
        triples.append([h, r, t, w])
    return np.array(triples)

triples = load_triples('/kaggle/input/cn15k-dataset/train.tsv')
soft_logic_triples = load_triples('/kaggle/input/cn15k-dataset/softlogic.tsv')
```

## Step 4: Soft Index

```python
n_soft_samples = 1
triple_indices = np.random.randint(0, soft_logic_triples.shape[0], size=n_soft_samples)
samples = soft_logic_triples[triple_indices]
soft_h_index, soft_r_index, soft_t_index, soft_w_index = (
    samples[:, 0].astype(int),
    samples[:, 1].astype(int),
    samples[:, 2].astype(int),
    samples[:, 3],
)
```

![image](https://github.com/hughiephan/UKGE/assets/16631121/d884e7cd-1445-4df0-b000-9f483d431ee5)

## Step 5: Gen and corrupt batch
```python
def gen_and_corrupt_batch(triples, batch_size, neg_per_positive, cons):
    N = len(cons)
    l = triples.shape[0]
    while True:
        np.random.shuffle(triples)
        for i in range(0, l, batch_size):
            batch = triples[i : i + batch_size, :]
                        
            if batch.shape[0] < batch_size:
                batch = np.concatenate((batch, triples[:batch_size - batch.shape[0]]), axis=0)

            h_batch, r_batch, t_batch, w_batch = (
                batch[:, 0].astype(int),
                batch[:, 1].astype(int),
                batch[:, 2].astype(int),
                batch[:, 3],
            )
            hrt_batch = batch[:, 0:3].astype(int)
            
            # Corrupt Batch
            neg_hn_batch = np.random.randint(0, N, size=(batch_size, neg_per_positive))
            neg_rel_hn_batch = np.tile(r_batch, (neg_per_positive, 1)).transpose()
            negt_batch = np.tile(t_batch, (neg_per_positive, 1)).transpose()
            negh_batch = np.tile(h_batch, (neg_per_positive, 1)).transpose()
            neg_rel_tn_batch = neg_rel_hn_batch
            neg_tn_batch = np.random.randint(0, N, size=(batch_size, neg_per_positive))
            
            yield (
                h_batch.astype(np.int64),
                r_batch.astype(np.int64),
                t_batch.astype(np.int64),
                w_batch.astype(np.float32),
                neg_hn_batch.astype(np.int64),
                neg_rel_hn_batch.astype(np.int64),
                negt_batch.astype(np.int64),
                negh_batch.astype(np.int64),
                neg_rel_tn_batch.astype(np.int64),
                neg_tn_batch.astype(np.int64),
            )
        if not forever:
            break
```

## Step 6: Define UKGE LOGI Model

Embedding dimensions `dim`, batch size `batch_size`, etc. It also sets default values for certain coefficients and variables used in the model.

Then defines the placeholders for input data: `_A_*` placeholders for indices of entities and relations, `_soft_*` placeholders for uncertain graph and PSL-related data and initializes trainable variables with `ht` for entity embeddings, and `r` for relation embeddings. Embeddings for positive and negative samples are looked up from the embedding matrices using `tf.nn.embedding_lookup`. With `main_loss` computes the main loss for the model. It uses the embeddings to calculate scores for positive and negative samples and computes a loss based on the difference between these scores and the expected scores `A_w` placeholder. While `psl_loss` computes the loss related to Probabilistic Soft Logic (PSL). It involves calculating a probability based on uncertain graph embeddings and then calculating the error against expected soft constraints. And the `opt` uses the Adam optimizer to minimize the combined loss `A_loss`, which includes both the main loss and the PSL loss. `train_op` applies the gradients computed by the optimizer to update the model parameters during training.

```python
class UKGE_LOGI(object):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg):
        self.num_rels = num_rels # Number of relations
        self.num_cons = num_cons # Number of ontologies
        self.dim = dim  # Dimension of both relation and ontology.
        self.neg_per_positive = neg_per_positive # Number of negative samples per (h,r,t)
        self.batch_size = batch_size
        self.p_psl = 0.2 # Coefficient
        self.p_neg = 1
        self.soft_size = 1
        self.prior_psl = 0

    def build(self):
        tf.reset_default_graph()
        with tf.variable_scope("graph", initializer=tf.truncated_normal_initializer(0, 0.3)):
            # Variables (matrix of embeddings/transformations)
            self._ht = ht = tf.get_variable(name='ht', shape=[self.num_cons, self.dim], dtype=tf.float32)
            self._r = r = tf.get_variable(name='r', shape=[self.num_rels, self.dim], dtype=tf.float32)
            self.A_h_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_h_index')
            self.A_r_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_r_index')
            self.A_t_index = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='A_t_index')
            # for uncertain graph
            self.A_w = tf.placeholder(dtype=tf.float32, shape=[self.batch_size], name='A_w')
            self.A_neg_hn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self.neg_per_positive), name='A_neg_hn_index')
            self.A_neg_rel_hn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self.neg_per_positive), name='A_neg_rel_hn_index')
            self.A_neg_t_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self.neg_per_positive), name='A_neg_t_index')
            self.A_neg_h_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self.neg_per_positive), name='A_neg_h_index')
            self.A_neg_rel_tn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self.neg_per_positive), name='A_neg_rel_tn_index')
            self.A_neg_tn_index = tf.placeholder(dtype=tf.int64, shape=(self.batch_size, self.neg_per_positive), name='A_neg_tn_index')
            # no normalization
            self.h_batch = tf.nn.embedding_lookup(ht, self.A_h_index)
            self.t_batch = tf.nn.embedding_lookup(ht, self.A_t_index)
            self.r_batch = tf.nn.embedding_lookup(r, self.A_r_index)
            self.neg_hn_con_batch = tf.nn.embedding_lookup(ht, self.A_neg_hn_index)
            self.neg_rel_hn_batch = tf.nn.embedding_lookup(r, self.A_neg_rel_hn_index)
            self.neg_t_con_batch = tf.nn.embedding_lookup(ht, self.A_neg_t_index)
            self.neg_h_con_batch = tf.nn.embedding_lookup(ht, self.A_neg_h_index)
            self.neg_rel_tn_batch = tf.nn.embedding_lookup(r, self.A_neg_rel_tn_index)
            self.neg_tn_con_batch = tf.nn.embedding_lookup(ht, self.A_neg_tn_index)
            # psl batches
            self.soft_h_index = tf.placeholder(dtype=tf.int64, shape=[self.soft_size], name='soft_h_index')
            self.soft_r_index = tf.placeholder(dtype=tf.int64, shape=[self.soft_size], name='soft_r_index')
            self.soft_t_index = tf.placeholder(dtype=tf.int64, shape=[self.soft_size], name='soft_t_index')
            # for uncertain graph and psl
            self.soft_w = tf.placeholder(dtype=tf.float32, shape=[self.soft_size], name='soft_w_lower_bound')
            self.softh_batch = tf.nn.embedding_lookup(ht, self.soft_h_index)
            self.softt_batch = tf.nn.embedding_lookup(ht, self.soft_t_index)
            self.softr_batch = tf.nn.embedding_lookup(r, self.soft_r_index)
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        ....
```

## Step 7: Compute Main Loss

Combine the H, T, and R together. Then we need to calculate the score of h, hn, and tn based on the probability. Finally, get the loss value based on the scores.

$$htr = \sum_{i=1}^{n} ( R_i \cdot (H_i \odot T_i ))$$

$$f_{prob_h} = \sigma(w * htr + b)$$

$$f_{score_h} = (f_{prob_h} - A_w)^2$$

$$f_{prob_{hn}} = \sigma ( w \cdot \sum_{i=1}^{n} ( negrel_{hn_i} \cdot \left(negcon_{hn_i} \cdot negcon_{t_i} \right) ) + b)$$

$$f_{score_{hn}} = \frac{1}{n} \sum_{i=1}^{n} (f_{prob_{hn_i}})^2$$

$$f_{prob_{tn}} = \sigma ( w \cdot \sum_{i=1}^{n} ( negrel_{tn_i} \cdot \left(negcon_{h_i} \cdot negcon_{tn_i} \right) ) + b)$$

$$f_{score_{tn}} = \frac{1}{n} \sum_{i=1}^{n} (f_{prob_{tn_i}})^2$$

$$loss_{main} = \frac{\sum (\frac{f_{score_{tn}} + f_{score_{hn}}}{2} \times p_{neg} + f_{score_h} )}{batchsize}$$

```python
        ...
        self.htr = tf.reduce_sum(tf.multiply(self.r_batch, tf.multiply(self.h_batch, self.t_batch, "element_wise_multiply"),"r_product"), 1)
        self.f_prob_h = tf.sigmoid(self.w * self.htr + self.b) # Logistic regression
        self.mse_pos = tf.square(tf.subtract(self.f_prob_h, self.A_w))
        self.f_prob_hn = tf.sigmoid(self.w * (tf.reduce_sum( tf.multiply(self.neg_rel_hn_batch, tf.multiply(self.neg_hn_con_batch, self.neg_t_con_batch)), 2)) + self.b)
        self.mse_neg = tf.reduce_mean(tf.square(self.f_prob_hn), 1)
        self.f_prob_tn = tf.sigmoid(self.w * (tf.reduce_sum(tf.multiply(self.neg_rel_tn_batch, tf.multiply(self.neg_h_con_batch, self.neg_tn_con_batch)), 2)) + self.b)
        self.f_score_tn = tf.reduce_mean(tf.square(self.f_prob_tn), 1)
        self.main_loss = (tf.reduce_sum(tf.add(tf.divide(tf.add(self.f_score_tn, self.mse_neg), 2) * self.p_neg, self.mse_pos))) / self.batch_size
        ...
```

## Step 8: Compute PSL Loss

$$prob_{psl} = \(\sigma ( w \cdot \sum_{i=1}^{n} ( \text{R}_i \cdot ( \text{H}_i \cdot \text{T}_i ) ) + b )\)$$

$$\text{psl-error-each} = ( \max ( 0, w + \text{prior-psl0} - prob_{psl}))^2$$ 

$$mse_{psl} = \frac{1}{N} \sum_{i=1}^{N} \text{psl-error-each}_i$$

$$loss_{psl} = mse_{psl} \cdot \text{p-psl}$$

With $\text{p-psl}$ is coefficient, and `prior_psl0` is just a constant 0, and psl-error-each seems like is derived from the Lukasiewicz t-norm

```python
        ....
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(tf.multiply(self.softr_batch, tf.multiply(self.softh_batch, self.softt_batch)), 1)+self.b)
        self.prior_psl0 = tf.constant(self.prior_psl, tf.float32)
        self.psl_error_each = tf.square(tf.maximum(self.soft_w + self.prior_psl0 - self.psl_prob, 0))
        self.psl_mse = tf.reduce_mean(self.psl_error_each)
        self.psl_loss = self.psl_mse * self.p_psl
        ...
```

## Step 9: Optimization

$$loss_{A} = loss_{main} + loss_{psl}$$

```python
        ...
        self.A_loss = tf.add(self.main_loss, self.psl_loss)
        self.lr = tf.placeholder(tf.float32)
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.gradient = self.opt.compute_gradients(self.A_loss) 
        self.train_op = self.opt.apply_gradients(self.gradient)
```

## Step 10: Model
```python
model = UKGE_LOGI(num_rels = len(rels),
                num_cons = len(cons),
                dim = dim,
                batch_size = batch_size, 
                neg_per_positive = 10, 
                p_neg = 1)
model.build()
```

## Step 11: Training

A session is created and started using `tf.Session()` and `Session.run` takes the operations we created and data to be fed as parameters and it returns the result. Only after running `tf.global_variables_initializer()` in a session will the variables hold the values you told them to hold.

```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
num_batch = triples.shape[0] // batch_size
print('Number of batches per epoch: %d' % num_batch)
train_losses = []
val_losses = []
for epoch in range(1, epochs + 1):
    generated_batch = gen_and_corrupt_batch(triples, batch_size, neg_per_positive, cons)
    epoch_loss = []
    for batch_id in range(num_batch):
        batch = next(generated_batch)
        A_h_index, A_r_index, A_t_index, A_w, A_neg_hn_index, A_neg_rel_hn_index, A_neg_t_index, A_neg_h_index, A_neg_rel_tn_index, A_neg_tn_index = batch
        _, gradient, A_loss, psl_mse, mse_pos, mse_neg, main_loss, psl_prob, psl_mse_each, _ = sess.run(
            [
                model.train_op, 
                model.gradient,
                model.A_loss,
                model.psl_mse, 
                model.mse_pos, 
                model.mse_neg,
                model.main_loss, 
                model.psl_prob, 
                model.psl_error_each,
                model.prior_psl0
            ],
            feed_dict={
                model.A_h_index: A_h_index,
                model.A_r_index: A_r_index,
                model.A_t_index: A_t_index,
                model.A_w: A_w,
                model.A_neg_hn_index: A_neg_hn_index,
                model.A_neg_rel_hn_index: A_neg_rel_hn_index,
                model.A_neg_t_index: A_neg_t_index,
                model.A_neg_h_index: A_neg_h_index,
                model.A_neg_rel_tn_index: A_neg_rel_tn_index,
                model.A_neg_tn_index: A_neg_tn_index,
                model.soft_h_index: soft_h_index,
                model.soft_r_index: soft_r_index,
                model.soft_t_index: soft_t_index,
                model.soft_w: soft_w_index, 
                model.lr: lr # Learning Rate
            })
        epoch_loss.append(A_loss)
        if ((batch_id + 1) % 50 == 0) or batch_id == num_batch - 1:
            print('process: %d / %d. Epoch %d' % (batch_id + 1, num_batch, epoch))
    this_total_loss = np.sum(epoch_loss) / len(epoch_loss)
    print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
```

Here's the result:

![image](https://github.com/hughiephan/UKGE/assets/16631121/f0b4d7f5-62c4-4755-b9a5-85f1e54fef43)

## Debugging (Optional)
```python
print("Total of soft_logic_triples:", soft_logic_triples.shape[0])
print("Pick indices by random triple_indices:", triple_indices)
print("Sample of", triple_indices[0],":", samples[0])
print("Round up Sample", triple_indices[0], ":", soft_h_index[0], soft_r_index[0], soft_t_index[0], soft_w_index[0])
```
