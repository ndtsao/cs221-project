#Jeff Sheng, CS 221
#Code to run RNNs on Wine Price
#Bidirectional RNN, GRU, and LSTM

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
import csv
import os
import re
import util
import numpy as np
import matplotlib.pyplot as plt

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
#from ner_model import NERModel
from defs import LBLS
#from utils.dataset import DataSet
from sklearn.metrics import confusion_matrix
#from nltk.tokenize import word_tokenize

from pdb import set_trace as t


class Config:

    #need a hidden size for the descriptions
    description_hidden_size = 50 #reduce to 50 next

    batch_size = 32
    embed_size = 200  # try a larger embedding as a possible fine-tuning

    max_length_description = 135  #the max of the reviews list is 135
    num_classes = 1
    epochs = 10
    lr = 0.0001 #lower by .0001
    dropout = 0.7
    final_state_size = 50


# This is starter code for only description and points prediction
# Full list (from 1-12): country, description, designation, points, price, province, region_1,
# region_2, taster_name, taster_twitter, title, variety, winery

def load_dataset():
    #take in each column from data as a list where indexes are the same review
    country, description, designation, points, price, province, region_1, region_2, taster_name, taster_twitter, title, variety, winery = \
        util.load("../data/wine-reviews/wine_cleaned_google-allprice.csv", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    description = [d.replace('.', '').replace(',', '').lower().split() for d in description]
    return description, price


# Jeff added this to pad sequences
def pad_sequences(data, max_length_description):
    ret = []
    lengths = []

    # Use this zero vector when padding sequences.
    zero_vector = 0  # * Config.n_features

    for description, price in data:
        # copy everything over
        copy_description = description[:]

        # fix the description
        description_len = len(description)
        if description_len >= max_length_description:
            copy_description = description[0:max_length_description]
        else:
            diff = max_length_description - description_len
            for _ in range(diff):
                copy_description.append(zero_vector)

        one_data_point = (copy_description, price)
        ret.append(one_data_point)
        lengths.append(min(description_len, max_length_description))

    return ret, lengths


#Starter code only pre-processes descriptions and price
def load_and_preprocess_data(debug=False):

    description, price = load_dataset()

    if debug:
        allData = [(description[_], float(price[_])) for _ in xrange(200)]

    else:
        allData = [(description[_], float(price[_])) for _ in xrange(len(description))]

    # choose specific description to comprise the training set

    np.random.seed(2017)
    np.random.shuffle(allData)

    train = allData[:int(len(allData) * 7 / 10)]
    dev = allData[int(len(allData) * 7 / 10):int(len(allData) * 9 / 10)]
    test = allData[int(len(allData) * 9 / 10):]

    helper = ModelHelper.build(allData)

    # ((the, drink, smells, like, this,, 85), (this wine is fruity, 90),.... )

    # now process all the input data.
    # train_data = helper.vectorize([x[0] for x in train])
    train_data = helper.vectorize(zip(*train)[0])
    dev_data = helper.vectorize(zip(*dev)[0])
    test_data = helper.vectorize(zip(*test)[0])

    train_final_data = zip(train_data, zip(*train)[1])
    dev_final_data = zip(dev_data, zip(*dev)[1])
    test_final_data = zip(test_data, zip(*test)[1])

    return helper, train_final_data, dev_final_data, test_final_data, train, dev, test

def generate_batches(data):
    # shuffle the data
    np.random.seed(2017)
    np.random.shuffle(data)

    # create batches
    batchNum = int(np.ceil(len(data) / float(Config.batch_size)))
    batches = []
    for i in range(batchNum):
        base = i * Config.batch_size
        batches.append(data[base:(min(base + Config.batch_size, len(data)))])
    return batches


if __name__ == "__main__":

    ######################################
    ##           prompt users           ##
    ######################################

    if len(sys.argv) < 2 or not (sys.argv[1] == "RNN" or sys.argv[1] == "GRU" or sys.argv[1] == "LSTM"):
        print "Error: must specify cell type!"
        exit()

    print 'What should the accuracy file name be?'
    accuracyFileName = raw_input()

    '''
    print 'What should the confusion matrix file name be?'
    cmFileName = raw_input()
    '''

    print 'What should the output file name be?'
    outputFileName = raw_input()


    ######################################
    ##           get the data           ##
    ######################################

    # load in the data
    debug = False
    if len(sys.argv) > 2 and sys.argv[2] == "debug":
        debug = True
    helper, train_final_data, dev_final_data, test_final_data, train, dev, test = load_and_preprocess_data(debug)
    pretrained_embeddings = load_embeddings(helper, vocabPath="../Vectors/gloveVocab.txt",
                                            vectorPath="../Vectors/glove.6B.200d.txt", wordFirst=True, embed_size=200)

    Config.embed_size = pretrained_embeddings.shape[1]

    # for later
    neverOpened_gold = True
    neverOpened_test = True

    ######################################
    ##           define graph           ##
    ######################################

    # define placeholders
    description_input_placeholder = tf.placeholder(tf.int32, (None, Config.max_length_description))
    labels_placeholder = tf.placeholder(tf.float32, (None, 1))  # do we need to specify num_classes
    description_lengths_placeholder = tf.placeholder(tf.int32, (None,))

    # dropout placeholder:
    dropout_placeholder = tf.placeholder(tf.float32, shape=())

    # create an embeddings variable
    embedding = tf.Variable(pretrained_embeddings, trainable=False)
    description_embedded_tensor = tf.nn.embedding_lookup(embedding, description_input_placeholder)

    # get the cell for descriptions
    if sys.argv[1] == "RNN":
        description_cell_fw = tf.nn.rnn_cell.BasicRNNCell(Config.description_hidden_size)
        description_cell_bw = tf.nn.rnn_cell.BasicRNNCell(Config.description_hidden_size)
    elif sys.argv[1] == "GRU":
        description_cell_fw = tf.nn.rnn_cell.GRUCell(Config.description_hidden_size)
        description_cell_bw = tf.nn.rnn_cell.GRUCell(Config.description_hidden_size)
    elif sys.argv[1] == "LSTM":
        description_cell_fw = tf.nn.rnn_cell.LSTMCell(Config.description_hidden_size)
        description_cell_bw = tf.nn.rnn_cell.LSTMCell(Config.description_hidden_size)

    # create the description cell
    with tf.variable_scope('description_cell'):
        description_output, description_states = tf.nn.bidirectional_dynamic_rnn(description_cell_fw, description_cell_bw,
                                                                                 description_embedded_tensor, dtype=tf.float32,
                                                                         sequence_length=description_lengths_placeholder)

    # first pass: add the tensors
    if sys.argv[1] != "LSTM":
        combined_description_state = description_states[0] + description_states[1]
    else:
        combined_description_state = description_states[0][1] + description_states[1][1]



    # define variables for multiplying
    W = tf.get_variable('W', (Config.description_hidden_size, Config.final_state_size),
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', Config.final_state_size, initializer=tf.zeros_initializer)

    # apply dropout rate here
    description_hidden = tf.nn.dropout(combined_description_state, dropout_placeholder)



    # compute

    last_hidden = tf.nn.relu(tf.matmul(description_hidden, W) + b)
    W2 = tf.get_variable('W2', (Config.final_state_size, Config.num_classes),
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', Config.num_classes, initializer=tf.zeros_initializer)
    output = tf.matmul(last_hidden, W2) + b2

    total_loss = tf.losses.mean_squared_error(labels_placeholder, output)
    loss = tf.reduce_mean(total_loss)

    # after getting loss we run the training optimizer and then pass this into sess.run() to call the model to train
    train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)



    ######################################
    ##               train              ##
    ######################################

    # graph already created

    # attribution: https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23
    # create a session
    session = tf.Session()

    # create the initialize operator and run it
    init = tf.global_variables_initializer()
    session.run(init)

    # loop over epoch count
    for epoch in range(Config.epochs):
        epoch_loss = 0

        # train_data needs to be a tuple of description vectors and price
        batches = generate_batches(train_final_data)

        # training loop
        batchCounter = 0
        for batch in batches:
            batchCounter = batchCounter + 1
            if batchCounter % 20 == 0:
                print batchCounter

            # the batch returned here is a tuple of description and price
            batch, lengths = pad_sequences(batch, Config.max_length_description)
            description_inputs_batch = np.array([x[0] for x in batch])
            description_lengths = np.array([x for x in lengths])
            labels_batch = np.array([x[1] for x in batch])
            labels_batch = np.expand_dims(labels_batch, axis=1)

            # create_feed_dict
            feed_dict = {description_input_placeholder: description_inputs_batch,
                         labels_placeholder: labels_batch,
                         description_lengths_placeholder: description_lengths,
                         dropout_placeholder: Config.dropout}
            if labels_batch is not None:
                feed_dict[labels_placeholder] = labels_batch

            # run session
            _, local_loss = session.run([train_op, loss], feed_dict=feed_dict)
            epoch_loss += local_loss

        # evaluation loop
        devBatches = generate_batches(dev_final_data)
        errors = 0.0

        for batch in devBatches:
            batch, lengths = pad_sequences(batch, Config.max_length_description)
            description_inputs_batch = np.array([x[0] for x in batch])
            description_lengths = np.array([x for x in lengths])
            labels_batch = np.array([x[1] for x in batch])
            labels_batch = np.expand_dims(labels_batch, axis=1)

            # create_feed_dict
            feed_dict = {description_input_placeholder: description_inputs_batch,
                         labels_placeholder: labels_batch,
                         description_lengths_placeholder: description_lengths,
                         dropout_placeholder: 1.0}

            predictions = session.run(output, feed_dict=feed_dict)

            errors += sum([(labels_batch[i] - predictions[i]) ** 2 for i in range(len(labels_batch))])

            # on last iteration, print out the results

        f = open('%s.txt' % accuracyFileName, 'a')
        print >> f, epoch
        print >> f, 'Training loss: ', epoch_loss / float(batchCounter)
        print >> f, 'Dev error: ', errors / float(len(dev_final_data))
        f.close()


        ######################################
        ##               test            ##
        ######################################


        ###  Final test  ###  Comment out until end ###
        if epoch == Config.epochs - 1:

            testBatches = generate_batches(test_final_data)
            errors = 0.0

            for batch in testBatches:
                batch, lengths = pad_sequences(batch, Config.max_length_description)
                description_inputs_batch = np.array([x[0] for x in batch])
                description_lengths = np.array([x for x in lengths])
                labels_batch = np.array([x[1] for x in batch])
                labels_batch = np.expand_dims(labels_batch, axis=1)

                # create_feed_dict
                feed_dict = {description_input_placeholder: description_inputs_batch,
                             labels_placeholder: labels_batch,
                             description_lengths_placeholder: description_lengths,
                             dropout_placeholder: 1.0}

                predictions = session.run(output, feed_dict=feed_dict)

                errors += sum([(labels_batch[i] - predictions[i]) ** 2 for i in range(len(labels_batch))])

                f = open('%s.txt' % outputFileName, 'a')

                print >> f, epoch
                print >> f, 'Training loss: ', epoch_loss / float(batchCounter)
                print >> f, 'Test error: ', errors / float(len(test))
                print >> f, 'Errors: ', errors
                print >> f, 'Length of Test: ', float(len(test))
                print >> f, 'Length of Dev ', float(len(dev))
                np.set_printoptions(threshold=np.inf)
                np.set_printoptions(linewidth=np.inf)
                np.set_printoptions(suppress=True)
                print >> f, 'Predicted for final were: ', predictions
                print >> f, 'Actual for final were', labels_batch
                f.close()

                with open(outputFileName + '-csv', 'a') as f:
                    writer = csv.writer(f)
                    for row in zip(labels_batch, predictions):
                        writer.writerow(row)
                    f.close()

