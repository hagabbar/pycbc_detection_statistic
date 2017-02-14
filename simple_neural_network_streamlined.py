#Simple Multilayer Neural Network to seperate pycbc injections from noise triggers
#Author: Hunter Gabbard
#Max Planck Insitute for Gravitational Physics
#How to use only one GPU device....export CUDA_VISIBLE_DEVICES="0" in command line prior to run
#How to run in the background from the command line...python simple_neural_network.py -d NSBH01_ifar0-1.hdf,NSBH02_ifar0-1.hdf >/dev/null 2>err.txt &

from __future__ import division
import argparse
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ActivityRegularization
from keras.optimizers import SGD
import sys
import os
from math import exp, log
import tensorflow as tf
from keras.callbacks import EarlyStopping
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import datetime

#Definition for loading in dataset parameters into variables and then combine into a numpy array
def load_back_data(data, params):
    trig_comb = {}
    params = ['marg_l','count','maxnewsnr','maxsnr','time','ratio_chirp','delT','delta_chirp','template_duration']
    h1 = h5py.File(data.split(',')[0], 'r')
    for fi in data:
        h1 = h5py.File(fi, 'r')
        if data.split(',')[0] == fi:
            for label in h1['H1'].keys():
                for key in params:
                    if label == key:
                        trig_comb[label] = np.asarray(h1['H1/%s' % label][:]).reshape((h1['H1/%s' % label].shape[0],1))
        else:
            for label in h1['H1'].keys():
                for key in params:
                    if label == key:
                        trig_comb[label+'_new'] = np.asarray(h1['H1/%s' % label][:]).reshape((h1['H1/%s' % label].shape[0],1))
                        trig_comb[label] = np.vstack((trig_comb[label], trig_comb[label+'_new']))
    return trig_comb

#Load CBC/noise triggers from multiple data sets
def load_inj_data(data, params):
    trig_comb = {}
    params = ['marg_l','count','maxnewsnr','maxsnr','time','ratio_chirp','delT','delta_chirp','template_duration']
    for fi in data:
        h1 = h5py.File(fi, 'r')
        if data.split(',')[0] == fi:
            for label in h1['H1'].keys():
                for key in params:
                    if label == key:
                        trig_comb[label] = np.asarray(h1['H1/%s' % label][:]).reshape((h1['H1/%s' % label].shape[0],1))
        else:
            for label in h1['H1'].keys():
                for key in params:
                    if label == key:
                        trig_comb[label+'_inj_new'] = np.asarray(h1['H1/%s' % label][:]).reshape((h1['H1/%s' % label].shape[0],1))
                        trig_comb[label+'_inj'] = np.vstack((trig_comb[label+'_inj'], trig_comb[label+'_inj_new']))

    return trig_comb



#Main function
def main(): 
    #Configure tensorflow to use gpu memory as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    #Use seed value of 32 for testing purposes
    np.random.seed(seed = 32)

    #Get current time for time stamp labels
    now = datetime.datetime.now()

    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
            help="path to input dataset")
    ap.add_argument("-b", "--back_dataset", required=True,
            help="path to one dataset file from each chunk you are running over")
    ap.add_argument("-o", "--output_dir", required=True,
            help="path to output directory")
    args = vars(ap.parse_args())

    data_files = args['dataset'].split(',')
    back_files = args['back_dataset'].split(',')
    out_dir = args['output_dir']

    trig_comb = load_back_data(back_files, )
    trig_comb = load_inj_data(data_files, )

    #Getting injection weights
    inj_weights_pre = []
    np.asarray(inj_weights_pre)
    dist_inj_mean = (dist_inj**2).mean()
    for idx in enumerate(delta_chirp_inj):
        idx = idx[0]
        inj_weights_pre.append((dist_inj[idx][0]**2)/dist_inj_mean)

    inj_weights = np.asarray(inj_weights_pre).reshape((delta_chirp_inj.shape[0],1))


if __name__ == '__main__':
    main()
