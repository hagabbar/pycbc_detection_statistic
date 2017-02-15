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
import unicodedata

#Definition for loading in dataset parameters into variables and then combine into a numpy array
def load_back_data(data, params):
    print 'loading background triggers'
    dict_comb = {}
    tmp_comb = {}
    h1 = h5py.File(data[0], 'r')
    for fi in data:
        h1 = h5py.File(fi, 'r')
        ifo = unicodedata.normalize('NFKD', h1.keys()[0]).encode('ascii','ignore')
        if data[0] == fi:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        dict_comb[label] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
        else:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        tmp_comb[label+'_new'] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                        dict_comb[label] = np.vstack((dict_comb[label], tmp_comb[label+'_new']))

    for idx,key in enumerate(dict_comb.keys()):
        if idx == 0:
            trig_comb = dict_comb[key]
        else:
            trig_comb = np.hstack((trig_comb,dict_comb[key]))

    return trig_comb, dict_comb

#Load CBC/noise triggers from multiple data sets
def load_inj_data(data, params, dict_comb):
    tmp_comb = {}
    print 'loading injections'
    for fi in data:
        h1 = h5py.File(fi, 'r')
        ifo = unicodedata.normalize('NFKD', h1.keys()[0]).encode('ascii','ignore')
        if data[0] == fi:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        dict_comb[label+'_inj'] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
        else:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        tmp_comb[label+'_inj_new'] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                        dict_comb[label+'_inj'] = np.vstack((dict_comb[label+'_inj'], tmp_comb[label+'_inj_new']))

    for idx,key in enumerate(dict_comb.keys()):
        if idx == 0:
            trig_comb = dict_comb[key]
        elif key == 'dist_inj':
            continue
        else:
            trig_comb = np.hstack((trig_comb,dict_comb[key]))

    return trig_comb, dict_comb

#Generate injection weights
def inj_weight_calc(inj_comb, dict_comb):
    print 'calculating injection weights'
    inj_weights_pre = []
    np.asarray(inj_weights_pre)
    dist_inj = dict_comb['dist_inj']
    dist_inj_mean = (dist_inj**2).mean()
    for idx in enumerate(delta_chirp_inj):
        idx = idx[0]
        inj_weights_pre.append((dist_inj[idx][0]**2)/dist_inj_mean)

    inj_weights = np.asarray(inj_weights_pre).reshape((dict_comb['delta_chirp_inj'].shape[0],1))

    return inj_weights

def orig_norm(back_trig, inj_trig, tt_split):
    print 'storing original trigger values prior to normalization'
    comb_all = np.vstack((back_trig, inj_trig))
    indices_trig = np.random.permutation(back_trig.shape[0])
    trig_train_idx, trig_test_idx = indices_trig[:int(back_trig.shape[0]*tt_split)], indices_trig[int(back_trig.shape[0]*tt_split):int(back_trig.shape[0])]
    trig_train_p, trig_test_p = back_trig[trig_train_idx,:], back_trig[trig_test_idx,:]
    indices_inj = np.random.permutation(inj_trig.shape[0])
    inj_train_idx, inj_test_idx = indices_inj[:int(inj_trig.shape[0]*tt_split)], indices_inj[int(inj_trig.shape[0]*tt_split):]
    inj_train_p, inj_test_p = inj_trig[inj_train_idx,:], inj_trig[inj_test_idx,:]
    train_data_p = np.vstack((trig_train_p, inj_train_p))
    test_data_p = np.vstack((trig_test_p, inj_test_p))

    return train_data_p, test_data_p, comb_all

def normalize(comb_all):
    print 'normalizing features'
    marg_l = ((np.log(comb_all[:,0]) - np.log(comb_all[:,0]).mean())/np.log(comb_all[:,0]).max()).reshape((comb_all.shape[0],1))
    count = ((comb_all[:,1] - comb_all[:,1].mean())/comb_all[:,1].max()).reshape((comb_all.shape[0],1))
    maxnewsnr = ((np.log(comb_all[:,2]) - np.log(comb_all[:,2]).mean())/np.log(comb_all[:,2]).max()).reshape((comb_all.shape[0],1))
    maxsnr = ((np.log(comb_all[:,3]) - np.log(comb_all[:,3]).mean())/np.log(comb_all[:,3]).max()).reshape((comb_all.shape[0],1))
    ratio_chirp = ((np.log(comb_all[:,4]) - np.log(comb_all[:,4]).mean())/np.log(comb_all[:,4]).max()).reshape((comb_all.shape[0],1))
    delT = ((comb_all[:,5] - comb_all[:,5].mean())/comb_all[:,5].max()).reshape((comb_all.shape[0],1))
    tmp_dur = ((np.log(comb_all[:,6]) - np.log(comb_all[:,6]).mean())/np.log(comb_all[:,6]).max()).reshape((comb_all.shape[0],1))
    trig_comb = np.hstack((marg_l[0:trig_comb.shape[0]],count[0:trig_comb.shape[0]],maxnewsnr[0:trig_comb.shape[0]],maxsnr[0:trig_comb.shape[0]],ratio_chirp[0:trig_comb.shape[0]],delT[0:trig_comb.shape[0]],tmp_dur[0:trig_comb.shape[0]]))
    inj_comb = np.hstack((marg_l[trig_comb.shape[0]:],count[trig_comb.shape[0]:],maxnewsnr[trig_comb.shape[0]:],maxsnr[trig_comb.shape[0]:],ratio_chirp[trig_comb.shape[0]:],delT[trig_comb.shape[0]:],tmp_dur[trig_comb.shape[0]:]))
    comb_all = np.vstack((trig_comb, inj_comb)) 

    return trig_comb, inj_comb, comb_all

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
    ap.add_argument("-t", "--train_perc", required=True,
            help="Percentage of triggers you want to train. Remaining percentage will be set aside for testing")
    args = vars(ap.parse_args())

    #Initializing parameters
    data_files = args['dataset'].split(',')
    back_files = args['back_dataset'].split(',')
    out_dir = args['output_dir']
    back_params = ['marg_l','count','maxnewsnr','maxsnr','time','ratio_chirp','delT','delta_chirp','template_duration']
    inj_params = ['marg_l','count','maxnewsnr','maxsnr','time','ratio_chirp','delT','delta_chirp','dist_inj','template_duration']
    tt_split = float(args['train_perc'])

    #Downloading background and injection triggers
    back_trig, dict_comb = load_back_data(back_files, back_params)
    inj_trig, dict_comb = load_inj_data(data_files, inj_params, dict_comb)
   
    #Getting injection weights for later use in neural network training process
    inj_weights = inj_weight_calc(inj_trig, dict_comb)

    #Storing original trigger feature values prior to normalization
    train_data_p, test_data_p, comb_all = orig_norm(back_trig, inj_trig, tt_split)    

    #Normalizin features from zero to one
    trig_comb, inj_comb, comb_all = normalize(comb_all)


    
if __name__ == '__main__':
    back_trig, inj_trig = main()
