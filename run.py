"""
=== usage
python run.py -m rect --data cn15k --lr 0.01 --batch_size 300
"""

# This is a test commit
import sys
import os
import datetime
import argparse
from os.path import join
from src import param
from src.data import Data
from src.trainer import Trainer
from src.utils import ModelList

# Commandline
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ppi5k', help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")
parser.add_argument("--verbose", help="print detailed info for debugging", action="store_true")
parser.add_argument('-m', '--model', type=str, default='rect', help="choose model ('logi' or 'rect'). default: rect")
parser.add_argument('-d', '--dim', type=int, default=128, help="set dimension. default: 128")
parser.add_argument('--epoch', type=int, default=100, help="set number of epochs. default: 100")
parser.add_argument('--lr', type=float, default=0.001, help="set learning rate. default: 0.001")
parser.add_argument('--batch_size', type=int, default=1024, help="set batch size. default: 1024")
parser.add_argument('--n_neg', type=int, default=10, help="Number of negative samples per (h,r,t). default: 10")
parser.add_argument('--save_freq', type=int, default=10, help="how often (how many epochs) to run validation and save tf models. default: 10")
parser.add_argument('--models_dir', type=str, default='./trained_models', help="the dir path where you store trained models. A new directory will be created inside it.") # Model will be trained, validated, and saved in './trained_models'
parser.add_argument('--reg_scale', type=float, default=0.0005, help="The scale for regularizer (lambda). Default 0.0005") # Regularizer coefficient (lambda)
args = parser.parse_args()

# Parameters
param.verbose = args.verbose
param.data = args.data
param.model = ModelList(args.model)
param.n_epoch = args.epoch
param.learning_rate = args.lr
param.batch_size = args.batch_size
param.val_save_freq = args.save_freq  # The frequency to validate and save model
param.dim = args.dim  # default 128
param.neg_per_pos = args.n_neg  # Number of negative samples per (h,r,t). default 10.
param.reg_scale = args.reg_scale

# Configure path to save model
now = datetime.datetime.now()
date = '%02d%02d' % (now.month, now.day) # Two digits month/day
identifier = param.model.value + '_' + date
save_dir = join(args.models_dir, param.data, identifier)  # The directory where we store this model
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('Trained models will be stored in: ', save_dir)

# Load data
data_dir = join('./data', args.data)
file_train = join(data_dir, 'train.tsv')  # training data
file_val = join(data_dir, 'val.tsv')  # validation data
file_psl = join(data_dir, 'softlogic.tsv')  # probabilistic soft logic
print('file_psl: %s' % file_psl)
more_filt = [file_val, join(data_dir, 'test.tsv')]
print('Read train.tsv from', data_dir)
this_data = Data()
this_data.load_data(file_train=file_train, file_val=file_val, file_psl=file_psl)
for f in more_filt:
    this_data.record_more_data(f)
this_data.save_meta_table(save_dir)  # output: idx_concept.csv, idx_relation.csv

# Start training
trainer = Trainer()
trainer.build(this_data, save_dir)
ht_embedding, r_embedding = trainer.train(epochs=param.n_epoch, save_every_epoch=param.val_save_freq, lr=param.learning_rate, data_dir=param.data_dir()) 