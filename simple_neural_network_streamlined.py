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
    back = {}
    tmp_comb = {}
    h1 = h5py.File(data[0], 'r')
    for fi in data:
        h1 = h5py.File(fi, 'r')
        ifo = unicodedata.normalize('NFKD', h1.keys()[0]).encode('ascii','ignore')
        if data[0] == fi:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        if key == 'delta_chirp' or key == 'time':
                            dict_comb[label] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                        else:
                            dict_comb[label] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                            back[label] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
        else:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        if key == 'delta_chirp' or key == 'time':
                            tmp_comb[label+'_new'] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                            dict_comb[label] = np.vstack((dict_comb[label], tmp_comb[label+'_new']))
                        else:                   
                            tmp_comb[label+'_new'] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                            back[label] = np.vstack((back[label], tmp_comb[label+'_new']))
                            dict_comb[label] = np.vstack((dict_comb[label], tmp_comb[label+'_new']))

    for idx,key in enumerate(params):
        print params
        if key == 'delta_chirp' or key == 'time':
            continue
        elif idx == 0:
            back_comb = back[key]
       
        else:
            back_comb = np.hstack((back_comb,back[key]))
    return back_comb, dict_comb

#Load CBC/noise triggers from multiple data sets
def load_inj_data(data, params, dict_comb):
    tmp_comb = {}
    inj = {}
    print 'loading injections'
    for fi in data:
        h1 = h5py.File(fi, 'r')
        ifo = unicodedata.normalize('NFKD', h1.keys()[0]).encode('ascii','ignore')
        if data[0] == fi:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        if key == 'delta_chirp_inj' or key == 'time_inj' or key == 'dist_inj':
                            dict_comb[label] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                        else:
                            dict_comb[label] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                            inj[label] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
        else:
            for label in h1['%s' % ifo].keys():
                for key in params:
                    if label == key:
                        if key == 'delta_chirp_inj' or key == 'time_inj' or key == 'dist_inj':
                            tmp_comb[label+'_new'] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                            dict_comb[label] = np.vstack((dict_comb[label], tmp_comb[label+'_new']))
                        else:
                            tmp_comb[label+'_new'] = np.asarray(h1['%s/%s' % (ifo,label)][:]).reshape((h1['%s/%s' % (ifo,label)].shape[0],1))
                            inj[label] = np.vstack((inj[label], tmp_comb[label+'_new']))
                        dict_comb[label] = np.vstack((dict_comb[label], tmp_comb[label+'_new']))

    for idx,key in enumerate(params):
        print params
        if key  == 'delta_chirp_inj' or key == 'time_inj' or key == 'dist_inj':
            continue
        elif idx == 0:
            inj_comb = inj[key]
        else:
            inj_comb = np.hstack((inj_comb,inj[key]))
    return inj_comb, dict_comb

#Generate injection weights
def inj_weight_calc(dict_comb):
    print 'calculating injection weights'
    inj_weights_pre = []
    np.asarray(inj_weights_pre)
    dist_inj = dict_comb['dist_inj']
    dist_inj_mean = (dist_inj**2).mean()
    for idx in enumerate(dict_comb['delta_chirp_inj']):
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

def sep(trig_comb,inj_comb,indices_trig,tt_split, inj_weights):
    print 'seperating into training/testing sets'
    trig_train_idx, trig_test_idx = indices_trig[:int(trig_comb.shape[0]*tt_split)], indices_trig[int(trig_comb.shape[0]*tt_split):int(trig_comb.shape[0])]
    trig_train, trig_test = trig_comb[trig_train_idx,:], trig_comb[trig_test_idx,:]
    indices_inj = np.random.permutation(inj_comb.shape[0])
    inj_train_idx, inj_test_idx = indices_inj[:int(inj_comb.shape[0]*tt_split)], indices_inj[int(inj_comb.shape[0]*tt_split):]
    inj_train_weight, inj_test_weight = inj_weights[inj_train_idx,:], inj_weights[inj_test_idx,:]
    inj_train, inj_test = inj_comb[inj_train_idx,:], inj_comb[inj_test_idx,:]
    train_data = np.vstack((trig_train, inj_train))
    test_data = np.vstack((trig_test, inj_test))
    train_data = train_data
    test_data = test_data

    return train_data, test_data, trig_test, inj_test, inj_test_weight, inj_train_weight

def normalize(trig_comb,comb_all):
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

def costco_label_maker(back_trig, inj_trig, tt_perc):
    print 'bought a label maker for my nn, it\'s pretty nice'
    #making labels (zero is noise, one is injection)
    c_zero = np.zeros((back_trig.shape[0],1))
    c_z_train = c_zero[:int(back_trig.shape[0]*tt_perc)]
    c_z_test = c_zero[int(back_trig.shape[0]*tt_perc):int(back_trig.shape[0])]
    c_ones = np.ones((int(inj_trig.shape[0]),1))
    c_o_train = c_ones[:int(inj_trig.shape[0]*tt_perc)]
    c_o_test = c_ones[int(inj_trig.shape[0]*tt_perc):int(inj_trig.shape[0])]
    lab_train = np.vstack((c_z_train,c_o_train))
    lab_test = np.vstack((c_z_test,c_o_test))
    labels_all = np.vstack((c_zero,c_ones))
 
    return lab_train, lab_test, labels_all

def samp_weight(trig_comb,inj_comb,inj_train_weight,inj_test_weight):
    print 'making sample weights vector'
    trig_weights = np.zeros((trig_comb.shape[0],1))
    trig_weights.fill(1/((trig_comb.shape[0])/(inj_comb.shape[0])))
    trig_w_train = trig_weights[:int(trig_comb.shape[0]*.7)]
    trig_w_test = trig_weights[int(trig_comb.shape[0]*.7):]
    train_weights = 100.*np.vstack((trig_w_train,inj_train_weight)).flatten()
    test_weights = 100.*np.vstack((trig_w_test,inj_test_weight)).flatten()

    return train_weights, test_weights

def the_machine(trig_comb, nb_epoch, batch_size, train_weights, test_weights, train_data, test_data, lab_train, lab_test, out_dir, now):
    print 'It\'s is alive!!!'
    model = Sequential()
    act = keras.layers.advanced_activations.ELU(alpha=1.0)                         #LeakyReLU(alpha=0.1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.add(Dense(10, input_dim=trig_comb.shape[1]))
    act
    model.add(Dense(7))
    act
    model.add(Dense(3))
    act
    model.add(Dense(3))
    act
    model.add(Dense(3))
    act
    model.add(Dense(3))
    act

    model.add(Dense(1, init='normal', activation='sigmoid'))

    #Compiling model
    print("[INFO] compiling model...")
    sgd = SGD(lr=0.05)
    model.compile(loss="binary_crossentropy", optimizer='rmsprop',
            metrics=["accuracy","binary_crossentropy"], class_mode='binary')
   
    #model.fit(train_data, lab_train, nb_epoch=1, batch_size=32, sample_weight=train_weights, shuffle=True, show_accuracy=True)
    hist = model.fit(train_data, lab_train,
                        nb_epoch=nb_epoch, batch_size=batch_size,    #66000
                        sample_weight=train_weights,
                        validation_data=(test_data,lab_test,test_weights),
                        shuffle=True, show_accuracy=True)
    print(hist.history) 

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    eval_results = model.evaluate(test_data, lab_test,
                                        sample_weight=test_weights,
                                            batch_size=32, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(eval_results[0],
            eval_results[1] * 100))
    #Saving prediction probabilities to a variable
    res_pre = model.predict(test_data)

    #Printing summary of model parameters
    model.summary()

    #Saving model to hdf file for later use
    os.makedirs('%s/run_%s' % (out_dir,now))
    os.makedirs('%s/run_%s/colored_plots' % (out_dir,now))
    os.makedirs('%s/run_%s/histograms' % (out_dir,now))
    model.save('%s/run_%s/nn_model.hdf' % (out_dir,now))
    np.save('%s/run_%s/hist.npy' % (out_dir,now), hist.history)

    return res_pre, eval_results, hist

#Function to compute ROC curve for both newsnr and some other score value
def ROC_inj_and_newsnr(trig_test,test_data,inj_test_weight,inj_test,lab_test,out_dir,now):
    print 'generating ROC curve plot'
    n_noise = len(trig_test)
    pred_prob = model.predict_proba(test_data, batch_size=32).T[0]
    prob_sort_noise = pred_prob[pred_prob[0:n_noise].argsort()][::-1]
    prob_sort_inj = pred_prob[n_noise:][pred_prob[n_noise:].argsort()][::-1]
    prob_sort_injWeight = inj_test_weight.T[0][pred_prob[n_noise:].argsort()][::-1]
    prob_sort_injNewsnr = inj_test[:,2][pred_prob[n_noise:].argsort()][::-1]
    newsnr_sort_noiseNewsnr = trig_test[:,2][trig_test[:,2][0:].argsort()][::-1]
    newsnr_sort_injNewsnr = inj_test[:,2][inj_test[:,2][0:].argsort()][::-1]
    newsnr_sort_injWeight = inj_test_weight.T[0][inj_test[:,2][0:].argsort()][::-1]
    pred_class = model.predict_classes(test_data)
    class_sort = pred_class[pred_prob[:].argsort()][::-1]
    orig_test_labels = lab_test[pred_prob[:].argsort()][::-1]


    #Initialize variables/arrays
    FAP = []
    ROC_w_sum = []
    ROC_newsnr_sum = []
    ROC_newsnr = []
    np.array(FAP)
    np.array(ROC_w_sum)
    np.array(ROC_newsnr_sum)
    np.array(ROC_newsnr)

    for idx in range(len(noise_param)):
        #Calculate false alarm probability value
        FAP.append((float(idx+1))/len(noise_param))

        #Compute sum
        w_sum = prob_sort_injWeight[prob_sort_inj >= prob_sort_noise[idx]].sum()
        newsnr_sum = newsnr_sort_injWeight[newsnr_sort_injNewsnr >= newsnr_sort_noiseNewsnr[idx]].sum()

        #Append
        ROC_w_sum.append(w_sum)
        ROC_newsnr_sum.append(newsnr_sum)

        #Normalize ROC y axis
        ROC_w_sum = np.asarray(ROC_w_sum)
        ROC_w_sum *= (1.0/ROC_w_sum.max())
        ROC_newsnr_sum = np.asarray(ROC_newsnr_sum)
        ROC_newsnr_sum *= (1.0/ROC_newsnr_sum.max())
        
    #Plot ROC Curve
    pl.plot(FAP,ROC_w_sum,label='NN Score')
    pl.plot(FAP,ROC_newsnr_sum,label='New SNR')
    pl.legend(frameon=True)
    pl.title('ROC Curve')
    pl.xlabel('False Alarm Probability')
    pl.ylabel('Weighted Sum')
    pl.xscale('log')
    pl.savefig('%s/run_%s/ROC_curve.png' % (out_dir,now))
    pl.close()

    return ROC_w_sum, ROC_newsnr_sum, FAP, pred_prob

#Function to compute ROC cruve given any weight and score. Not currently used, but could be used later if desired
def ROC(inj_weight, inj_param, noise_param, out_dir, now):
    print 'generating ROC curve plot'
    #Assert that length of inj weights and injection parameters are the same
    assert len(inj_weight) == len(inj_param)
    
    #Initialize variables/arrays
    ROC_value = 0
    FAP = []
    np.array(FAP)
    ROC_sum = []
    np.array(ROC_sum)

    for idx in range(len(noise_param)):
        #Calculate false alarm probability value
        FAP.append((float(idx+1))/len(noise_param))

        #Compute sum
        ROC_value = inj_weight[inj_param >= noise_param[idx]].sum()

        #Append
        ROC_sum.append(ROC_value)

        #Normalize ROC y axis
        ROC_sum = np.asarray(ROC_sum)
        ROC_sum *= (1.0/ROC_sum.max())

    #Plot ROC Curve
    pl.plot(FAP,ROC_sum,label='Score')
    pl.legend(frameon=True)
    pl.title('ROC Curve')
    pl.xlabel('False Alarm Probability')
    pl.ylabel('Weighted Sum')
    pl.xscale('log')
    pl.savefig('%s/run_%s/ROC_curve.png' % (out_dir,now))
    pl.close()


    return ROC_sum, FAP

def main_plotter(out_dir, now, test_data_p, params, back_test, hist):
    #Loss vs. Epoch
    print 'plotting loss vs. epoch'
    pl.plot(hist.history['loss'])
    pl.title('Loss vs. Epoch')
    pl.xlabel('Epoch')
    pl.ylabel('Loss')
    pl.yscale('log')
    pl.savefig('%s/run_%s/loss_vs_epoch.png' % (out_dir,now))
    pl.close()

    #Accuracy vs. Epoch
    print 'plotting accuracy vs. epoch'
    pl.plot(hist.history['acc'])
    pl.title('Accuracy vs. Epoch')
    pl.xlabel('Epoch')
    pl.ylabel('Accuracy')
    pl.savefig('%s/run_%s/acc_vs_epoch.png' % (out_dir,now))
    pl.close()

    for idx,label in enumerate(params):
        print('plotting score vs. %s' % label)
        pl.scatter(test_data_p[0:n_noise,idx],pred_prob[0:n_noise],marker="o",label='background')
        pl.scatter(test_data_p[n_noise:,idx],pred_prob[n_noise:],marker="^",label='injection')
        pl.legend(frameon=True)
        pl.title('Score vs. %s' % label)
        pl.ylabel('Score')
        pl.xlabel('%s' % label)
        pl.savefig('%s/run_%s/score_vs_%s.png' % (out_dir,now,label))
        pl.close()

        print('plotting %s histogram' % label)
        pl.hist(test_data_p[0:n_noise,idx], 50, label='background')
        pl.hist(test_data_p[n_noise:,idx], 50, label='injection')
        pl.legend(frameon=True)
        pl.title('%s histogram' % label)
        pl.xlabel('%s' % label)
        pl.savefig('%s/run_%s/histograms/%s.png' % (out_dir,now,label))
        pl.close()

        for idx2,label2 in enumerate(params):
             print('plotting %s vs. %s' (label2,label))
             pl.scatter(test_data_p[0:n_noise,idx],test_data_p[0:n_noise,idx2],c=pred_prob[0:n_noise],marker="o",label='background')
             pl.scatter(test_data_p[n_noise:,idx],test_data_p[n_noise:,idx2],c=pred_prob[n_noise:],marker="^",label='injection')
             pl.legend(frameon=True)
             pl.title('%s vs. %s' % (label2,label))
             pl.xlabel('%s' % label)
             pl.ylabel('%s' % label2)
             pl.colorbar()
             pl.savefig('%s/run_%s/colored_plots/%s_vs_%s.png' % (out_dir,now,idx2,idx))
             pl.close() 

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
    ap.add_argument("-e", "--nb_epoch", required=True,
            help="Number of Epochs")
    ap.add_argument("-bs", "--batch_size", required=True,
            help="Batch size for the training process (e.g. number of samples to introduce to network at any given epoch)")
    args = vars(ap.parse_args())

    #Initializing parameters
    data_files = args['dataset'].split(',')
    back_files = args['back_dataset'].split(',')
    out_dir = args['output_dir']
    now = datetime.datetime.now()       #Get current time for time stamp labels
    back_params = ['marg_l','count','maxnewsnr','maxsnr','ratio_chirp','delT','template_duration','delta_chirp','time']
    inj_params = ['marg_l_inj','count_inj','maxnewsnr_inj','maxsnr_inj','ratio_chirp_inj','delT_inj','template_duration_inj','dist_inj','delta_chirp_inj','time_inj']
    tt_split = float(args['train_perc'])
    nb_epoch = int(args['nb_epoch'])
    batch_size = int(args['batch_size'])

    #Downloading background and injection triggers
    back_trig, dict_comb = load_back_data(back_files, back_params)
    inj_trig, dict_comb = load_inj_data(data_files, inj_params, dict_comb)

    #Getting injection weights for later use in neural network training process
    inj_weights = inj_weight_calc(dict_comb)

    #Storing original trigger feature values prior to normalization
    train_data_p, test_data_p, comb_all = orig_norm(back_trig, inj_trig, tt_split)    

    #Normalizing features from zero to one
    back_trig, inj_trig, comb_all = normalize(back_trig, comb_all)

    #Randomizing the order of the background triggers
    indices_trig = np.random.permutation(back_trig.shape[0])

    #Seperating into training/testing sets
    train_data, test_data, back_test, inj_test, inj_test_weight, inj_train_weight = sep(back_trig, inj_trig, indices_trig, tt_split, inj_weights)

    #making labels (zero is noise, one is injection)...better label maker than one you could buy at costco in my opinion
    lab_train, lab_test, labels_all = costco_label_maker(back_trig, inj_trig, tt_split)

    #Creating sample weights vector
    train_weights, test_weights = samp_weight(back_trig, inj_trig, inj_train_weight, inj_test_weight)

    #training/testing on deep neural network
    res_pre, eval_results, hist = the_machine(back_trig, nb_epoch, batch_size, train_weights, test_weights, train_data, test_data, lab_train, lab_test, out_dir, now)
    
    #Compute the ROC curve
    ROC_w_sum, ROC_newsnr_sum, FAP, pred_prob = ROC(back_test,test_data,inj_test_weight,inj_test,lab_test)

    #Score/histogram plots
    main_plotter(out_dir, now, test_data_p, params, back_test, hist)

    #Write data to an hdf file
    with h5py.File('%s/run_%s/nn_data.hdf' % (out_dir,now), 'w') as hf:
        hf.create_dataset('FAP', data=FAP)
        hf.create_dataset('ROC_w_sum', data=ROC_w_sum)
        hf.create_dataset('pred_prob', data=pred_prob)
        hf.create_dataset('test_data', data=test_data)
        hf.create_dataset('train_data', data=train_data)
        hf.create_dataset('ROC_newsnr_sum', data=ROC_newsnr_sum)

    print 'and...presto! You\'re done!'
if __name__ == '__main__':
    main()
