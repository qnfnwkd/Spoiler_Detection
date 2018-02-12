#! /usr/bin/env python

import os
import sys
import time
import tensorflow as tf
import numpy as np
import utils


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_file", "../../dataset/spoilers/train.txt", "Training data set")
tf.flags.DEFINE_string("test_file", "../../dataset/spoilers/test.txt", "Testing data set")
tf.flags.DEFINE_string("wordembedding", "../../dataset/spoilers/wordembedding.npy", "Pre trained Word Embedding")
tf.flags.DEFINE_string("word_list", "../../dataset/spoilers/word_list.txt", "Word List in Dataset")

# Model Hyperparameters
tf.flags.DEFINE_integer("wordembedding_dim", 300, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("genreembedding_dim", 25, "Dimensionality of genre")
tf.flags.DEFINE_integer("hidden_state", 100, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("relu_1", 100, "Dimensionality of word embedding")
tf.flags.DEFINE_string("filter_sizes", "1,2", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 50, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda (default: 0.0)")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 1024, "Number of training epochs")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learing rate")
tf.flags.DEFINE_float("dropout_rate",0.5, "Dropout rate")

# Tensorflow Option Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
train, test = utils.load_data(FLAGS.train_file, FLAGS.test_file)

# Build Dictionary
print("Build Dictionary...")
word2id, id2word, genre2id, id2genre = utils.build_dic(train, test, FLAGS.word_list)

# Convert Data to Index
print("Converting Data...")
train, test, maximum_document_length, maximum_genre_length = utils.converting(train, test, word2id, genre2id)

print("Loading Pre-trained Word Embedding...")
with open(FLAGS.wordembedding) as f:
    _wordembedding = np.load(f)

print("word dict size: "+str(len(word2id)))
print("genre dict size: "+str(len(genre2id)))
print("Train/Test: {:d}/{:d}".format(len(train),len(test)))
print("==================================================================================")

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        
        # ===========================
        # input layer
        # ===========================

        dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
        input_content = tf.placeholder(tf.int32, [None, maximum_document_length], name="input_content")
        input_genre = tf.placeholder(tf.int32, [None, maximum_genre_length], name="input_genre")
        input_spoiler = tf.placeholder(tf.int32, [None], name="input_spoiler")

        # keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        with tf.device("/cpu:0"):
            # word embedding matrix
            word_embedding = tf.Variable(tf.truncated_normal([len(word2id), FLAGS.wordembedding_dim]), trainable=True, name="word_embedding")
            embedding_placeholder = tf.placeholder(tf.float32, [len(word2id), FLAGS.wordembedding_dim])
            embedding_init = word_embedding.assign(embedding_placeholder)
            embedded_words = tf.nn.embedding_lookup(word_embedding, input_content)
            
            # genre embedding matrix
            genre_embedding = tf.Variable(tf.truncated_normal([len(genre2id), FLAGS.genreembedding_dim]), trainable=True, name="genre_embedding")
            embedded_genres = tf.nn.embedding_lookup(genre_embedding, input_genre)

        # identity matrix for using a softmax_cross_entropy loss function
        identity_matrix = tf.constant(np.identity(2))
        corrected_spoiler = tf.nn.embedding_lookup(identity_matrix, input_spoiler)

        # ===========================
        # textual RNN layer
        # ===========================

        # for dynamic RNN class to handle inputs of various lengths
        input_content_length = tf.count_nonzero(input_content,1)
        input_genre_length = tf.count_nonzero(input_genre,1)

        # Bi-directional
        gru_f = tf.contrib.rnn.GRUCell(FLAGS.hidden_state)
        gru_b = tf.contrib.rnn.GRUCell(FLAGS.hidden_state)
        gru_f = tf.nn.rnn_cell.DropoutWrapper(gru_f, dropout_rate)
        gru_b = tf.nn.rnn_cell.DropoutWrapper(gru_b, dropout_rate)

        with tf.variable_scope("content"):
            output, h_n = tf.nn.bidirectional_dynamic_rnn(gru_f, gru_b, embedded_words, dtype=tf.float32, sequence_length = input_content_length)
        h_n = tf.concat(h_n,1)
        output = tf.concat(output, 2)

        # Genre Encoding Layer
        # for convolution expanded layer
        embedded_genres_expanded = tf.expand_dims(embedded_genres, -1)

        pooled_outputs = []
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, FLAGS.genreembedding_dim, 1, FLAGS.num_filters]
                convolution_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="convolution_w")
                convolution_b = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_filters]), name="convolution_b")
                conv = tf.nn.conv2d(embedded_genres_expanded, convolution_w, strides=[1,1,1,1], padding="VALID", name="conv")
            
            # Apply nonlinearity
            convolution_h = tf.nn.relu(tf.nn.bias_add(conv, convolution_b), name="relu")
            
            # Mapooling
            pooled = tf.nn.max_pool(convolution_h, ksize=[1, maximum_genre_length - filter_size + 1, 1, 1], strides=[1,1,1,1], padding="VALID", name="pool")
            pooled_outputs.append(pooled)
        
        num_filters_total = FLAGS.num_filters * len(filter_sizes)
        pool_concat = tf.concat(pooled_outputs, 3)

        # Genre feature
        genre_encoder = tf.reshape(pool_concat, [-1, num_filters_total])

        # Attention Layer
        ## Match dimension of genre feature with rnn hidden state
        attention_w = tf.get_variable("attention_w", shape=[2*FLAGS.hidden_state, num_filters_total], initializer=tf.contrib.layers.xavier_initializer())
        attention_b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name="attention_b")
        attention_hidden = tf.reshape(tf.nn.xw_plus_b(tf.reshape(output, [-1, 2*FLAGS.hidden_state]), attention_w, attention_b), [-1, maximum_document_length, 2*FLAGS.num_filters])

        ## Coumpute the Attention Weight
        attention_weight = tf.reduce_sum(tf.multiply(attention_hidden, tf.reshape(genre_encoder, [-1,1,num_filters_total])),2, keep_dims=True)
        attention_weight = tf.nn.softmax(attention_weight, dim=1)
        attention_mask = tf.sequence_mask(input_content_length, maximum_document_length, dtype=tf.float32)
        content_encoder = tf.reduce_sum(tf.multiply(tf.multiply(output, tf.reshape(attention_mask,[-1,maximum_document_length,1])), attention_weight), 1)

        ff1_w = tf.get_variable("ff1_w", shape=[4*FLAGS.hidden_state, FLAGS.relu_1], initializer=tf.contrib.layers.xavier_initializer())
        ff1_b = tf.Variable(tf.constant(0.1, shape=[FLAGS.relu_1]), name="ff_b")

        ## Concatenate the Last Hidden State
        output = tf.concat([content_encoder, h_n],1)
        output = tf.nn.dropout(output, dropout_rate)
        output = tf.nn.xw_plus_b(output, ff1_w, ff1_b)
        output = tf.nn.relu(output)

        ff2_w = tf.get_variable("ff2_w", shape=[FLAGS.relu_1, 2], initializer=tf.contrib.layers.xavier_initializer())
        ff2_b = tf.Variable(tf.constant(0.1, shape=[2]), name="ff_b")
        
        output = tf.nn.dropout(output, dropout_rate)
        output = tf.nn.xw_plus_b(output, ff2_w, ff2_b)

        _, rank = tf.nn.top_k(output, k=1)

        l2_loss += tf.nn.l2_loss(ff1_w)
        l2_loss += tf.nn.l2_loss(ff1_b)
        l2_loss += tf.nn.l2_loss(ff2_w)
        l2_loss += tf.nn.l2_loss(ff2_b)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=corrected_spoiler)
        loss = tf.reduce_mean(losses) + FLAGS.l2_reg_lambda * l2_loss

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        sess.run(embedding_init, feed_dict={embedding_placeholder: _wordembedding})

        def train_step(content_batch, genre_batch, spoiler_batch):
            feed_dict = {input_content:content_batch, input_genre:genre_batch, input_spoiler:spoiler_batch, dropout_rate:FLAGS.dropout_rate}
            _loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            return _loss

        def validation_step(content_batch, genre_batch):
            feed_dict = {input_content:content_batch, input_genre:genre_batch, dropout_rate:1.0}
            _rank, _attention_weight = sess.run([rank,attention_weight], feed_dict=feed_dict)
            return _rank, _attention_weight


        best = [0, 0.0, 0.0, 0.0, 0.0]
        print("Training..\n")
        for i in range(FLAGS.num_epochs):
            # Training
            _loss = .0
            step = 0
            train_batches = utils.batch_iter(train,FLAGS.batch_size)
            num_batches = int(len(train)/FLAGS.batch_size) + 1
            for batch in train_batches:
                content_batch, genre_batch, spoiler_batch = batch
                _loss+=train_step(content_batch, genre_batch, spoiler_batch)
                step+=1
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                print("Process Context Layer Epoch: [{}/{}] Batch: [{}/{}]".format(i+1, FLAGS.num_epochs, step, num_batches))

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print("Process Context Layer Epoch: [{}/{}] Loss: {}\n".format(i+1, FLAGS.num_epochs, _loss))

            # Evaluation
            atten = []
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print("Evaluation at epoch #{:d}...".format(i+1))
            count = 0
            retrieved = 0
            relevant = 0
            step = 0
            accuracy = 0
            validation_batches = utils.batch_iter(train,FLAGS.batch_size)
            num_batches = int(len(train)/FLAGS.batch_size) + 1
            for batch in validation_batches:
                content_batch, genre_batch, spoiler_batch = batch
                _rank, _atten_w=validation_step(content_batch, genre_batch)
                for j in xrange(len(_rank)):
                    correct_spoiler = spoiler_batch[j]
                    predicted_spoiler = _rank[j][0]
                    
                    if correct_spoiler == predicted_spoiler:
                        accuracy+=1
                    if correct_spoiler == 1:
                        relevant+=1
                    if predicted_spoiler == 1:
                        retrieved+=1
                    if correct_spoiler == 1 and predicted_spoiler == 1:
                        count+=1
                        atten.append((content_batch[j], genre_batch[j], _atten_w[j]))
                step+=1
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                print("Process Context Layer Epoch: [{}/{}] Batch: [{}/{}]".format(i+1, FLAGS.num_epochs, step, num_batches))

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            
            if retrieved == 0 or count == 0:
                print("Does not Retrieved\n======================================\n")
                continue
            
            accuracy = accuracy/float(len(train))
            precision = count/float(retrieved)
            recall = count/float(relevant)
            f1 = 2*precision*recall/(precision+recall)
            if best[4] < accuracy:
                best[0] = i+1
                best[1] = precision
                best[2] = recall
                best[3] = f1
                best[4] = accuracy
                _atten = []
                for line in atten:
                    _content = [id2word[j] for j in line[0] if j != 0]
                    _genre = [id2genre[j] for j in line[1] if j != 0]
                    _atten_w = [j[0] for j in line[2][:len(_content)]]
                    _atten.append("{}\t{}\t{}".format(" ".join(_genre), " ".join(_content), " ".join([str(a) for a in _atten_w])))

                with open("attention_result.txt", "w") as f:
                    f.write("\n".join(_atten))

            print("Validation Result Epoch: [{}/{}] Score [P/R/F/A]: [{}/{}/{}/{}]".format(i+1, FLAGS.num_epochs, precision, recall, f1, accuracy))
            print("Best Result Epoch #{} Score: [{}/{}/{}/{}]".format(best[0], best[1], best[2], best[3], best[4]))
            print("======================================\n")            
