#Simple Multilayer Neural Network to separate pycbc injections from noise triggers
#Authors: Hunter Gabbard, Tom Dent
#Max Planck Institute for Gravitational Physics
#How to use only one GPU device....export CUDA_VISIBLE_DEVICES="0" in command line prior to run
#How to run in the background from the command line...python simple_neural_network.py -d NSBH01_ifar0-1.hdf,NSBH02_ifar0-1.hdf >/dev/null 2>err.txt &


from __future__ import division
import argparse
import keras
from keras.models import Sequential
import numpy as np
import h5py
from keras.layers import Dense, Activation, Dropout, GaussianDropout, ActivityRegularization
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
import os, sys, shutil
from math import exp, log
import tensorflow as tf
from keras.callbacks import EarlyStopping
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import datetime
import unicodedata


# Load in dataset parameters and combine into a numpy array
def load_back_data(data, params):
    print 'loading background triggers'
    dict_comb = {}
    back = {}
    tmp_comb = {}
    for i, fi in enumerate(data):
        print fi
        h1 = h5py.File(fi, 'r')
        ifo = unicodedata.normalize('NFKD', h1.keys()[0]).encode('ascii','ignore')
        # set up original array
        if i == 0:
            for key in params:
                if key == 'time':
                    dict_comb[key] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                elif key == 'chirp_m':
                    dict_comb[key] = np.asarray(h1['%s/%s' % (ifo,key)][:,0]).reshape((h1['%s/%s' % (ifo,'maxsnr')].shape[0],1))
                else:
                    dict_comb[key] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                    back[key] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
        else:
            # stack on additional data
            for key in params:
                if key == 'time':
                    tmp_comb[key+'_new'] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                    dict_comb[key] = np.vstack((dict_comb[key], tmp_comb[key+'_new']))
                elif key == 'chirp_m':
                    tmp_comb[key+'_new'] = np.asarray(h1['%s/%s' % (ifo,key)][:,0]).reshape((h1['%s/%s' % (ifo,'maxsnr')].shape[0],1))
                    dict_comb[key] = np.vstack((dict_comb[key], tmp_comb[key+'_new']))
                else:
                    tmp_comb[key+'_new'] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                    back[key] = np.vstack((back[key], tmp_comb[key+'_new']))
                    dict_comb[key] = np.vstack((dict_comb[key], tmp_comb[key+'_new']))

    #Saving background trigger gps time
    back_time = dict_comb['time'].reshape((dict_comb['maxsnr'].shape[0],1))

    for idx, key in enumerate(params):
        print key
        if key == 'time' or key == 'chirp_m':
            continue
        elif idx == 0:
            back_comb = back[key]
        else:
            back_comb = np.hstack((back_comb, back[key]))

    return back_comb, dict_comb, back_time

#Load injection triggers from multiple data sets
def load_inj_data(data, params, dict_comb, weight):
    tmp_comb = {}
    inj = {}
    print 'loading injections'
    for i, fi in enumerate(data):
        print fi
        h1 = h5py.File(fi, 'r')
        ifo = unicodedata.normalize('NFKD', h1.keys()[0]).encode('ascii','ignore')
        if i == 0:
            for key in params:
                # treat inj distance differently
                if key == 'dist_inj' or key == 'opt_snr' or key == 'time_inj':
                    dict_comb[key] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                elif key == 'chirp_m_inj':
                    dict_comb[key] = np.asarray(h1['%s/%s' % (ifo,key)][:,0]).reshape((h1['%s/%s' % (ifo,'maxsnr_inj')].shape[0],1))
                else:
                    dict_comb[key] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                    inj[key] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
        else:
            for key in params:
                if key == 'dist_inj' or key == 'opt_snr' or key == 'time_inj':  # distance goes into dict_comb but not into inj_comb
                    tmp_comb[key+'_new'] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                    dict_comb[key] = np.vstack((dict_comb[key], tmp_comb[key+'_new']))
                elif key == 'chirp_m_inj':
                    tmp_comb[key+'_new'] = np.asarray(h1['%s/%s' % (ifo,key)][:,0]).reshape((h1['%s/%s' % (ifo,'maxsnr_inj')].shape[0],1))
                    dict_comb[key] = np.vstack((dict_comb[key], tmp_comb[key+'_new']))
                else:
                    tmp_comb[key+'_new'] = np.asarray(h1['%s/%s' % (ifo,key)][:]).reshape((h1['%s/%s' % (ifo,key)].shape[0],1))
                    inj[key] = np.vstack((inj[key], tmp_comb[key+'_new']))
                    dict_comb[key] = np.vstack((dict_comb[key], tmp_comb[key+'_new']))
    
    #Create opt_snr mask so as to remove those injections with optimal snr of zero
    mask = np.invert(np.isinf(dict_comb['opt_snr']))
    
    #Saving injection gps time and chirp mass
    if weight == 'optimal_snr':
        inj_time = dict_comb['time_inj'][mask].reshape((dict_comb['maxsnr_inj'][mask].shape[0],1))
        inj_chirpm = dict_comb['chirp_m_inj'][mask].reshape((dict_comb['maxsnr_inj'][mask].shape[0],1))
    elif weight == 'distance':
        inj_time = dict_comb['time_inj'].reshape((dict_comb['maxsnr_inj'].shape[0],1))
        inj_chirpm = dict_comb['chirp_m_inj'].reshape((dict_comb['maxsnr_inj'].shape[0],1))

    for idx, key in enumerate(params):
        print key
        if key == 'dist_inj' or key == 'opt_snr' or key == 'time_inj' or key == 'chirp_m_inj':
            continue
        elif idx == 0 and weight == 'optimal_snr':
            inj_comb = inj[key][mask].reshape((dict_comb['maxsnr_inj'][mask].shape[0],1))  
        elif idx == 0 and weight == 'distance':
            inj_comb = inj[key]
        elif idx > 0 and weight == 'optimal_snr':
            inj_comb = np.hstack((inj_comb, inj[key][mask].reshape((dict_comb['maxsnr_inj'][mask].shape[0],1)))) 
        elif idx > 0 and weight == 'distance':
            inj_comb = np.hstack((inj_comb, inj[key]))
    return inj_comb, dict_comb, inj_time, inj_chirpm

#Generate source distance injection weights
#def inj_weight_calc(dict_comb):
#    print 'calculating injection weights'
#    inj_weights_pre = []
#    dist_inj = dict_comb['dist_inj']
#    w_mean = (dist_inj**2).mean()
#    for idx, val in enumerate(dict_comb['maxsnr_inj']):
#        inj_weights_pre.append((dist_inj[idx][0] ** 2) / w_mean)

#    return np.asarray(inj_weights_pre).reshape((dict_comb['maxsnr_inj'].shape[0],1))

#Generate optimal snr injection weights
def inj_weight_calc(dict_comb, weight, inj_chirpm):
    print 'calculating injection weights'
    
    if weight == 'optimal_snr':
        #Create opt_snr mask so as to remove those injections with optimal snr of zero
        mask = np.invert(np.isinf(dict_comb['opt_snr']))

        inj_weights_pre = []
        opt_snr = dict_comb['opt_snr']
        w_mean = opt_snr[mask].mean()
        for idx, val in enumerate(dict_comb['maxsnr_inj'][mask]):
             inj_weights_pre.append(opt_snr[mask][idx] / w_mean)
        inj_weights_pre = np.asarray(inj_weights_pre).reshape((dict_comb['maxsnr_inj'][mask].shape[0],1))
        
    elif weight == 'distance':
        inj_weights_pre = []
        dist_inj = dict_comb['dist_inj']
        inj_chirpm = dict_comb['chirp_m_inj']
        w_mean = ((inj_chirpm**(-5./3.))*(dist_inj**2)).mean()
        for idx, val in enumerate(dict_comb['maxsnr_inj']):
            inj_weights_pre.append(((inj_chirpm[idx]**(-5./3.))*(dist_inj[idx][0] ** 2)) / w_mean)  
        inj_weights_pre = np.asarray(inj_weights_pre).reshape((dict_comb['maxsnr_inj'].shape[0],1))  
        
    return inj_weights_pre

def orig_norm(bg_trig, inj_trig, tt_split, perm):
    print 'storing original trigger values prior to normalization'
    n_bg, n_inj = bg_trig.shape[0], inj_trig.shape[0]
    comb_all = np.vstack((bg_trig, inj_trig))
    indices_bg = perm[0]
    bg_train_idx, bg_test_idx = indices_bg[:int(n_bg * tt_split)], indices_bg[int(n_bg * tt_split):int(n_bg)] # WHY LAST INDEX NEEDED?
    bg_train_p, bg_test_p = bg_trig[bg_train_idx,:], bg_trig[bg_test_idx,:]
    indices_inj = perm[1]
    inj_train_idx, inj_test_idx = indices_inj[:int(n_inj * tt_split)], indices_inj[int(n_inj * tt_split):]
    inj_train_p, inj_test_p = inj_trig[inj_train_idx,:], inj_trig[inj_test_idx,:]
    train_data_p = np.vstack((bg_train_p, inj_train_p))
    test_data_p = np.vstack((bg_test_p, inj_test_p))

    return train_data_p, test_data_p, comb_all

def sep(bg_comb, inj_comb, indices_bg, tt_split, inj_weights, comb_all, back_time, inj_time, perm):
    print 'separating into training/testing sets'
    n_bg, n_inj = bg_comb.shape[0], inj_comb.shape[0]
    bg_train_idx, bg_test_idx = perm[0][:int(n_bg * tt_split)], perm[0][int(n_bg * tt_split):int(n_bg)]  # WHY LAST INDEX NEEDED?
    bg_train, bg_test = bg_comb[bg_train_idx,:], bg_comb[bg_test_idx,:]
    indices_inj = perm[1]
    inj_train_idx, inj_test_idx = indices_inj[:int(n_inj * tt_split)], indices_inj[int(n_inj * tt_split):]
    inj_train_weight, inj_test_weight = inj_weights[inj_train_idx,:], inj_weights[inj_test_idx,:]
    inj_train, inj_test = inj_comb[inj_train_idx,:], inj_comb[inj_test_idx,:]
    train_data = np.vstack((bg_train, inj_train))
    test_data = np.vstack((bg_test, inj_test))

    #Seperating gps times into testing and training sets
    bg_train_times, bg_test_times = back_time[bg_train_idx,:], back_time[bg_test_idx,:]
    inj_train_times, inj_test_times = inj_time[inj_train_idx,:], inj_time[inj_test_idx,:]
    train_times = np.vstack((bg_train_times, inj_train_times))
    test_times = np.vstack((bg_test_times, inj_test_times))

    print comb_all

    return train_data, test_data, bg_test, inj_test, inj_test_weight, inj_train_weight, train_times, test_times

def normalize(comb_all, pre_proc_log, n_bg):
    print 'normalizing (and logging some) features'
    for idx in range(0, comb_all.shape[1]):
       vals = comb_all[:,idx]

       if pre_proc_log[idx] == True:
           # hack +1 to allow us to take log of count_out
           if vals.min() <= 0:
               print 'Adding 1 to a feature where some values are 0 or negative, index', idx
               vals = vals + 1
           vals = np.log(vals)

       normvals = ((vals - vals.mean()) / vals.max()).reshape((comb_all.shape[0], 1))

       if idx == 0:
           tmp_bg_comb = normvals[0:n_bg]
           tmp_inj_comb = normvals[n_bg:]
       else:
           tmp_bg_comb = np.hstack((tmp_bg_comb, normvals[0:n_bg])) 
           tmp_inj_comb = np.hstack((tmp_inj_comb, normvals[n_bg:])) 

    comb_all = np.vstack((tmp_bg_comb, tmp_inj_comb)) 
    return tmp_bg_comb, tmp_inj_comb, comb_all

def label_maker(back_trig, inj_trig, tt_perc):
    # zero is noise, one is injection
    n_bg, n_inj = back_trig.shape[0], inj_trig.shape[0]
    c_zero = np.zeros((n_bg, 1))
    c_z_train = c_zero[:int(n_bg * tt_perc)]
    c_z_test = c_zero[int(n_bg * tt_perc):int(n_bg)]  # why int(n_bg)?
    c_ones = np.ones((n_inj, 1))
    c_o_train = c_ones[:int(n_inj * tt_perc)]
    c_o_test = c_ones[int(n_inj * tt_perc):int(n_inj)] # why int()?
    lab_train = np.vstack((c_z_train, c_o_train))
    lab_test = np.vstack((c_z_test, c_o_test))
    labels_all = np.vstack((c_zero, c_ones))
 
    return lab_train, lab_test, labels_all

def samp_weight(bg_comb, inj_comb, inj_w_train, inj_w_test, tt_perc):
    print 'making sample weights vector'
    bg_weights = np.zeros((bg_comb.shape[0], 1))
    bg_weights.fill(30. * inj_comb.shape[0] / bg_comb.shape[0])
    bg_w_train = bg_weights[:int(bg_comb.shape[0]*tt_perc)]
    bg_w_test = bg_weights[int(bg_comb.shape[0]*tt_perc):]
    train_weights = np.vstack((bg_w_train, inj_w_train)).flatten()
    test_weights = np.vstack((bg_w_test, inj_w_test)).flatten()

    return train_weights, test_weights, bg_weights

def the_machine(args, n_features, train_weights, test_weights, train_data, test_data, lab_train, lab_test, out_dir, now):
    model = Sequential()
    drop_rate = args.dropout_fraction
    ret_rate = 1 - drop_rate
    act = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)
    dro = GaussianDropout(drop_rate)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=2)  # not used?

    model.add(Dense(7, input_dim=n_features)) #Add in Dense 7
    act
    model.add(BatchNormalization())
    model.add(GaussianDropout(0.1))

    model.add(Dense(int(7./ret_rate)))
    act
    model.add(BatchNormalization())
    model.add(dro)

#    model.add(Dense(int(7./ret_rate)))
#    act
#    model.add(BatchNormalization())
#    model.add(dro)

#    model.add(Dense(int(7./ret_rate)))
#    act
#    model.add(BatchNormalization())
#    model.add(dro)

#    model.add(Dense(int(7./ret_rate)))
#    act
#    model.add(BatchNormalization())
#    model.add(dro)

    #Additional hidden lay testing
#    model.add(Dense(int(7./ret_rate)))
#    act
#    model.add(BatchNormalization())
#    model.add(dro)

#    model.add(Dense(int(7./ret_rate)))
#    act
#    model.add(BatchNormalization())
#    model.add(dro)

#    model.add(Dense(int(7./ret_rate)))
#    act
#    model.add(BatchNormalization())
#    model.add(dro)


    model.add(Dense(1, init='normal', activation='sigmoid'))

    #Compiling model
    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=SGD(lr=args.learning_rate, momentum=0.8, decay=args.learning_rate/args.n_epoch, nesterov=True),
            metrics=["accuracy","binary_crossentropy"]) #, class_mode='binary')
   
    hist = model.fit(train_data, lab_train,
                     nb_epoch=args.n_epoch, batch_size=args.batch_size,
                     sample_weight=train_weights,
                     validation_data=(test_data,lab_test,test_weights),
                     shuffle=True)
    #print(hist.history) 

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    eval_results = model.evaluate(test_data, lab_test,
                                  sample_weight=test_weights,
                                  batch_size=args.batch_size, verbose=1)
    #print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(eval_results[0],
    #        eval_results[1] * 100))
    #Saving prediction probabilities to a variable
    res_pre = model.predict(test_data)

    #Printing summary of model parameters
    model.summary()

    #Saving model to hdf file for later use
    model.save('%s/run_%s/nn_model.hdf' % (out_dir,now))
    np.save('%s/run_%s/hist.npy' % (out_dir,now), hist.history)

    return res_pre, eval_results, hist, model

#Function to compute ROC curve for both newsnr and some other score value
def ROC_inj_and_newsnr(batch_size,trig_test,test_data,inj_test_weight,inj_test,lab_test,out_dir,now,model,train_times,test_times,test_data_p):
    print 'generating ROC curve plot' 

    n_noise = len(trig_test)
    pred_prob = model.predict_proba(test_data, batch_size=batch_size).T[0]
    prob_sort_noise = pred_prob[pred_prob[0:n_noise].argsort()][::-1]
    prob_sort_inj = pred_prob[n_noise:][pred_prob[n_noise:].argsort()][::-1]
    prob_sort_injWeight = inj_test_weight.T[0][pred_prob[n_noise:].argsort()][::-1]
    prob_sort_injNewsnr = inj_test[:,2][pred_prob[n_noise:].argsort()][::-1]
    newsnr_sort_noiseNewsnr = trig_test[:,2][trig_test[:,2][0:].argsort()][::-1]
    newsnr_sort_injNewsnr = inj_test[:,2][inj_test[:,2][0:].argsort()][::-1]
    newsnr_sort_injWeight = inj_test_weight.T[0][inj_test[:,2][0:].argsort()][::-1]
    pred_class = model.predict_classes(test_data, batch_size=batch_size)
    class_sort = pred_class[pred_prob[:].argsort()][::-1]
    orig_test_labels = lab_test[pred_prob[:].argsort()][::-1]

    #Sorting gps triggers/injections
    #First indice is loudest, last indice is quitest.  
    bg_time_sorted = test_times[0:n_noise][pred_prob[0:n_noise].argsort()][::-1]
    inj_time_sorted = test_times[n_noise:][pred_prob[n_noise:].argsort()][::-1]

    bg_sorted = test_data_p[0:n_noise,:][pred_prob[0:n_noise].argsort()][::-1]
    inj_sorted = test_data_p[n_noise:,:][pred_prob[n_noise:].argsort()][::-1]

    fmt = '{:<8}{:<20}{}'
    nameList= ['count_in', 'count_out', 'maxnewsnr', 'maxsnr', 'ratio_chirp', 'delT', 'template_duration']

    print '\nFive highest ranked background events by neural network ...'
    print(fmt.format('', 'GPS Time', '  -- count_in -- count_out -- maxnewsnr -- maxsnr -- ratio_chirp delT -- template_duration'))

    #Storing ten highest ranked background events by neural network into txt file
    f = open('%s/run_%s/highest_ranked_trigs.txt' % (out_dir,now), 'a')
    print >>f, '\nTen highest ranked background events by neural network ...'
    print >>f, fmt.format('', 'GPS Time', '  -- count_in -- count_out -- maxnewsnr -- maxsnr -- ratio_chirp delT -- template_duration')

    for i, (gpstime, features) in enumerate(zip(bg_time_sorted[0:10], bg_sorted[0:10,:])):
        print(fmt.format(i, gpstime, features))    
        print >>f, fmt.format(i,gpstime, features)

    #Initialize variables/arrays
    w_sum = 0
    newsnr_sum = 0
    FAP = []
    ROC_w_sum = []
    ROC_newsnr_sum = []

    for idx in range(n_noise):
        #Calculate false alarm probability value
        FAP.append((float(idx+1))/n_noise)

        #Compute sum
        w_sum = prob_sort_injWeight[prob_sort_inj >= prob_sort_noise[idx]].sum()
        newsnr_sum = newsnr_sort_injWeight[newsnr_sort_injNewsnr >= newsnr_sort_noiseNewsnr[idx]].sum()
        ROC_w_sum.append(w_sum)
        ROC_newsnr_sum.append(newsnr_sum)

    #Normalize ROC y axis
    ROC_w_sum = np.asarray(ROC_w_sum)
    ROC_w_sum *= (1./ROC_w_sum.max())
    ROC_newsnr_sum = np.asarray(ROC_newsnr_sum)
    ROC_newsnr_sum *= (1./ROC_newsnr_sum.max())
        
    #Plot ROC Curve
    pl.figure()
    pl.plot(FAP,ROC_w_sum,label='NN Score')
    pl.plot(FAP,ROC_newsnr_sum,label='New SNR')
    pl.ylim(ymax=1.)
    pl.legend(frameon=True, loc='lower right')
    #pl.title('ROC Curve')
    pl.xlabel('False alarm probability')
    pl.ylabel('Relative detection rate')
    pl.xscale('log')
    pl.savefig('%s/run_%s/ROC_curve.png' % (out_dir,now))
    pl.close()

    return ROC_w_sum, ROC_newsnr_sum, FAP, pred_prob, prob_sort_noise, prob_sort_inj

#Function to compute ROC curve given any weight and score. Not currently used, but could be used later if desired
def ROC(inj_weight, inj_param, noise_param, out_dir, now):
    print 'generating ROC curve plot'
    
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
        ROC_sum *= (1./ROC_sum.max())

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


def feature_hists(run_num, out_dir, now, params, pre_proc_log, nn_train, nn_test, train_data, test_data):
    print 'plotting feature histograms'
    for idx, lab in enumerate(zip(params, pre_proc_log)):
        print lab[0]
        for data, noise_len, dtype in zip([train_data, test_data], [nn_train, nn_test], ['train', 'test']):
            pl.figure(run_num+idx)
            hist_1, bins_1 = np.histogram(data[0:noise_len,idx], bins=100, density=True)
            hist_2, bins_2 = np.histogram(data[noise_len:,idx], bins=100, density=True)
            # print lab[0], dtype, 'bg', bins_1.min(), bins_1.max(), 'inj', bins_2.min(), bins_2.max()
            width_1 = (bins_1[1] - bins_1[0])
            width_2 = (bins_2[1] - bins_2[0])
            center_1 = (bins_1[:-1] + bins_1[1:]) / 2
            center_2 = (bins_2[:-1] + bins_2[1:]) / 2
            pl.bar(center_1, hist_1, log=True, label='background',color='b', alpha=0.6, align='center', width=width_1)
            pl.bar(center_2, hist_2, log=True, label='injection', color='r', alpha=0.6, align='center', width=width_2)
            pl.ylim(ymin=1e-4)
            pl.legend(frameon=True)
            if lab[1]:
               # pl.title('log(%s) histogram' % lab[0])
                pl.xlabel('log(%s) [normalized]' % lab[0])
            else:
               # pl.title('%s histogram' % lab[0])
                pl.xlabel('%s [normalized]' % lab[0])
            pl.savefig('%s/run_%s/histograms/%s_%s.png' % (out_dir, now, lab[0], dtype))
            pl.close()


def main_plotter(prob_sort_noise, prob_sort_inj, run_num, out_dir, now, test_data_p, params, back_test, hist, pred_prob, pre_proc_log, train_times, test_times, dict_comb, bg_weights, inj_weights):

    print 'plotting training metrics'
    print hist.history.keys()
    for i, metric in enumerate(['loss', 'acc', 'binary_crossentropy']):
        mname = metric.replace('acc', 'accuracy')
        pl.figure(run_num+i)
        pl.plot(hist.history[metric], label='Training')
        pl.plot(hist.history['val_'+metric], label='Validation')
        pl.legend(frameon=True, loc='center right')
        pl.xlabel('Epoch')
        pl.ylabel(mname.replace('_', ' '))
        if metric == 'acc':
            pl.ylim(ymin=0.7)
        else:
            pl.ylim(ymax=3. * hist.history[metric][-1])
        pl.savefig('%s/run_%s/%s_vs_epoch.png' % (out_dir, now, mname[0:4]))
        pl.close()

    print 'plotting histograms of score values'
    pl.figure(run_num+2)
    numpy_hist_1, bins_1 = np.histogram(prob_sort_noise, bins=100, density=True)
    numpy_hist_2, bins_2 = np.histogram(prob_sort_inj, bins=100, density=True)
    width_1 = (bins_1[1] - bins_1[0])
    width_2 = (bins_2[1] - bins_2[0])
    center_1 = (bins_1[:-1] + bins_1[1:]) / 2
    center_2 = (bins_2[:-1] + bins_2[1:]) / 2
    pl.bar(center_1, numpy_hist_1, log=True, label='background',color='b', alpha=0.6, align='center', width=width_1)
    pl.bar(center_2, numpy_hist_2, log=True, label='injection', color='r', alpha=0.6, align='center', width=width_2)
    pl.ylim(ymin=1e-4)
    pl.legend(frameon=True)
    pl.savefig('%s/run_%s/score_hist.png' % (out_dir,now))
    pl.close()

    print 'plotting histogram of weights'
    pl.figure(run_num+2)
    numpy_hist_2, bins_2 = np.histogram(inj_weights, bins=100, density=True) 
    width_2 = (bins_2[1] - bins_2[0])
    center_2 = (bins_2[:-1] + bins_2[1:]) / 2
    pl.bar(center_2, numpy_hist_2, log=True, label='injection', color='r', alpha=0.4, align='center', width=width_2)
    pl.legend(frameon=True)
    pl.savefig('%s/run_%s/weights_hist.png' % (out_dir,now))
    pl.close()

    n_noise = len(back_test)
    #For weighting diagnostic purposes ONLY. Remove when done with diagnostics
    print 'plotting scatter loglog plot of duration as a funciton of chrip mass'
    tmp_dur_inj = dict_comb['template_duration_inj']
    tmp_dur = dict_comb['template_duration']
    inj_chirpm = dict_comb['chirp_m_inj']
    bg_chirpm = dict_comb['chirp_m']
    pl.figure(run_num+3)
    pl.scatter(np.log(tmp_dur), np.log(bg_chirpm), marker='o', s=8, c='k', edgecolor='none', label='background', alpha=0.3)
    pl.scatter(np.log(tmp_dur_inj), np.log(inj_chirpm), marker='^', s=10, c='r', edgecolor='none', label='injection', alpha=0.4)
    pl.legend(frameon=True)
    pl.title('Chirp Mass vs. Template Duration')
    pl.xlabel('log(Template Duration)')
    pl.ylabel('log(Chirp Mass)')
    pl.savefig('%s/run_%s/chirpm_vs_tempdur' % (out_dir,now))
    pl.close()


    for idx,lab in enumerate(zip(params,pre_proc_log)):
        print('plotting score vs. %s' % lab[0])
        pl.figure(run_num+idx)
        pl.scatter(test_data_p[0:n_noise,idx],pred_prob[0:n_noise],marker="o", s=8,c='k',edgecolor='none',label='background',alpha=0.3)
        pl.scatter(test_data_p[n_noise:,idx], pred_prob[n_noise:], marker="^",s=10,c='r',edgecolor='none',label='injection', alpha=0.4)
        pl.legend(frameon=True)
        if lab[1] == True:
            pl.title('Score vs. log(%s)' % lab[0])
            pl.xlabel('log(%s)' % lab[0])
        else:
            pl.title('Score vs. %s' % lab[0])
            pl.xlabel('%s' % lab[0])
        pl.ylabel('Score')
        pl.ylim(0,1)
        pl.savefig('%s/run_%s/score_vs_%s.png' % (out_dir,now,lab[0]))
        pl.close()

        for idx2,lab2 in enumerate(zip(params,pre_proc_log)):
             if lab[0] > lab2[0]:
                 print('plotting %s vs. %s' % (lab2[0], lab[0]))
                 pl.figure(run_num)
                 pl.scatter(test_data_p[0:n_noise,idx],test_data_p[0:n_noise,idx2],c=pred_prob[0:n_noise],marker="o",s=10,edgecolor='none',label='background',alpha=0.4)
                 pl.scatter(test_data_p[n_noise:,idx], test_data_p[n_noise:,idx2], c=pred_prob[n_noise:], marker="^",s=10,edgecolor='none',label='injection', alpha=0.4)
                 pl.legend(frameon=True)
                 if lab2[1] == True and lab[1] == True:
                     pl.title('log(%s) vs. log(%s)' % (lab2[0], lab[0]))
                     pl.xlabel('log(%s)' % lab[0])
                     pl.ylabel('log(%s)' % lab2[0])
                 elif lab2[1] == True and lab[1] == False:
                     pl.title('log(%s) vs. %s' % (lab2[0], lab[0]))
                     pl.xlabel('%s' % lab[0])
                     pl.ylabel('log(%s)' % lab2[0])
                 elif lab2[1] == False and lab[1] == True:
                     pl.title('%s vs. log(%s)' % (lab2[0], lab[0]))
                     pl.xlabel('log(%s)' % lab[0])
                     pl.ylabel('%s' % lab2[0])
                 elif lab2[1] == False and lab[1] == False:
                     pl.title('%s vs. %s' % (lab2[0], lab[0]))
                     pl.xlabel('%s' % lab[0])
                     pl.ylabel('%s' % lab2[0])
                 pl.colorbar()
                 pl.savefig('%s/run_%s/colored_plots/%s_vs_%s.png' % (out_dir,now,lab2[0],lab[0]))
                 pl.close()
             else:
                 continue


#Main function
def main():
    #For TESTING ONLY
    #shutil.rmtree('/home/hunter.gabbard/public_html/simple_neural_net/testing/classification/L1-all_of_O1_run/top_5_trigs/run_test3')

 
    #Configure tensorflow to use gpu memory as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    #don't #Use seed value of 32 for testing purposes
    #np.random.seed(seed = 32)

    #Get Current time
    cur_time = datetime.datetime.now()       #Get current time for time stamp labels

    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--inj-files", nargs='+',
        help="path to injection HDF files")
    ap.add_argument("-b", "--bg-files", nargs='+',
        help="path to background HDF files [currently, one inj file per chunk is used as source of bg data]")
    ap.add_argument("-o", "--output-dir", required=True,
        help="path to output directory")
    ap.add_argument("-t", "--train-perc", type=float, required=False, default=0.5,
        help="Fraction of triggers you want to train (between 0 and 1). Remaining triggers will be used for testing. Default 0.5")
    ap.add_argument("-e", "--n-epoch", type=int, required=False, default=100,
        help="Number of epochs. Default 100")
    ap.add_argument("-bs", "--batch-size", type=int, required=False, default=32,
        help="Batch size for the training process (number of samples to use in each gradient descent step). Default 32")
    ap.add_argument("--learning-rate", type=float, default=0.01,
        help="Learning rate. Default 0.01")
    ap.add_argument("--dropout-fraction", type=float, default=0.,
        help="Amount of Gaussian dropout noise to use in training. Default 0 (no noise)")
    ap.add_argument("-u", "--usertag", required=False, default=cur_time,
        help="label for given run")
    ap.add_argument("-r", "--run-number", type=int, required=False, default=0,
        help="If performing multiple runs on same machine, specify a unique number for each run (must be greater than zero)")
    ap.add_argument("-w", "--weight", required=False, type=str, default='distance',
        help="Choose a sample weighting scheme (e.g. optimal_snr or distance")
    args = ap.parse_args()

    #Initializing parameters
    out_dir = args.output_dir
    #now = datetime.datetime.now()       #Get current time for time stamp labels
    now = args.usertag
    os.makedirs('%s/run_%s' % (out_dir,now))  # Fail early if the dir already exists
    os.makedirs('%s/run_%s/colored_plots' % (out_dir,now))
    os.makedirs('%s/run_%s/histograms' % (out_dir,now))

    back_params = ['count_in', 'count_out', 'maxnewsnr', 'maxsnr', 'ratio_chirp', 'delT', 'template_duration', 'time', 'chirp_m']
    inj_params = ['count_in_inj', 'count_out_inj', 'maxnewsnr_inj', 'maxsnr_inj', 'ratio_chirp_inj', 'delT_inj', 'template_duration_inj', 'dist_inj', 'opt_snr', 'time_inj', 'chirp_m_inj']
    pre_proc_log = [True,True,True,True,True,False,True] #True means to take log of feature, False means don't take log of feature during pre-processing
    batch_size = args.batch_size
    run_num = args.run_number
    weight = args.weight

    #Downloading background and injection triggers
    bg_trig, dict_comb, back_time = load_back_data(args.bg_files, back_params)
    inj_trig, dict_comb, inj_time, inj_chirpm = load_inj_data(args.inj_files, inj_params, dict_comb, weight)

    #Getting injection weights for later use in neural network training process
    inj_weights = inj_weight_calc(dict_comb, weight, inj_chirpm)

    #Set random shuffles of bg and inj features once and for all
    indices_bg = np.random.permutation(bg_trig.shape[0])
    indices_inj = np.random.permutation(inj_trig.shape[0])
    perm = (indices_bg, indices_inj)

    #Storing original trigger feature values prior to normalization
    train_data_p, test_data_p, comb_all = orig_norm(bg_trig, inj_trig, args.train_perc, perm)

    #Normalizing feature means and ranges (and logging some of them)
    bg_trig, inj_trig, comb_all = normalize(comb_all, pre_proc_log, bg_trig.shape[0])
    print bg_trig.shape

    #Need bg indices for next function. Work that in...

    #Separating into training/testing sets
    train_data, test_data, back_test, inj_test, inj_w_test, inj_w_train, train_times, test_times = \
      sep(bg_trig, inj_trig, indices_bg, args.train_perc, inj_weights, comb_all, back_time, inj_time, perm)
    print train_data.shape

    #making labels (zero is noise, one is injection)
    lab_train, lab_test, labels_all = label_maker(bg_trig, inj_trig, args.train_perc)

    #Creating sample weights vector
    train_weights, test_weights, bg_weights = samp_weight(bg_trig, inj_trig, inj_w_train, inj_w_test, args.train_perc)

    #Plot histograms of features
    feature_hists(run_num, out_dir, now, back_params, pre_proc_log, sum(lab_train.flatten() == 0), len(back_test), train_data, test_data)

    #training/testing on neural network
    res_pre, eval_results, hist, model = the_machine(args, bg_trig.shape[1], train_weights, test_weights, train_data, test_data, lab_train, lab_test, out_dir, now)

    #Compute the ROC curve
    ROC_w_sum, ROC_newsnr_sum, FAP, pred_prob, prob_sort_noise, prob_sort_inj = ROC_inj_and_newsnr(batch_size, back_test, test_data, inj_w_test, inj_test, lab_test, out_dir, now, model, train_times, test_times, test_data_p)

    #Score/histogram plots
    main_plotter(prob_sort_noise, prob_sort_inj, run_num, out_dir, now, test_data, back_params, back_test, hist, pred_prob, pre_proc_log, train_times, test_times, dict_comb, bg_weights, inj_weights)

    #Write data to an hdf file
    with h5py.File('%s/run_%s/nn_data.hdf' % (out_dir,now), 'w') as hf:
        hf.create_dataset('FAP', data=FAP)
        hf.create_dataset('ROC_w_sum', data=ROC_w_sum)
        hf.create_dataset('pred_prob', data=pred_prob)
        hf.create_dataset('test_data', data=test_data)
        hf.create_dataset('train_data', data=train_data)
        hf.create_dataset('ROC_newsnr_sum', data=ROC_newsnr_sum)
        hf.create_dataset('inj_weights', data=inj_weights)

    print 'Done!'
if __name__ == '__main__':
    main()
