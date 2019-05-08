'''
Based on the official word2vec example given by tensorflow project.
'''
from __future__ import division, print_function, absolute_import

import collections
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

##
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
##
if len(sys.argv) != 2:
    print("specify mini_occ!")
    exit(0)
print("Train word2vec on partial domain words.")
# Training Parameters
learning_rate = 0.1
batch_size = 128
num_steps =  2000000 
display_step = 10000
eval_step = 200000 #change from 200000 to 100000

# Evaluation Parameters
eval_words = ['qualcomm', 'apple', 'microsoft', 'facebook'] ## add b in front of strings, Shibo 

# Word2Vec Parameters
embedding_size = 300 # Dimension of the embedding vector
max_vocabulary_size = 50000 # Total number of different words in the vocabulary
min_occurrence = int(sys.argv[1]) # Remove all words that does not appears at least n times

skip_window = 3 # How many words to consider left and right
num_skips = 2 # How many times to reuse an input to generate a label
num_sampled = 64 # Number of negative examples to sample

'''
df = pd.read_csv(os.path.abspath('../process/sp500_token.csv'))
listLabel = df.nGram.values.tolist() # Shibo
listLabel = [l.lower() for l in listLabel]
'''
with open("stopwords.txt", 'r') as f:
    stopwords = f.read().split('\n')
    stopwords = set(stopwords)

with open("../../data/finance/ptb.train.txt", 'r') as f: # Shibo
    text_words = f.read().split()
    text_words = [t for t in text_words if t not in stopwords]
# Build the dictionary and replace rare words with UNK token
count = [('<unk>', -1)]

temp = collections.Counter(text_words)
unk_count = temp['<unk>']
print(unk_count)
del temp['<unk>']
# Retrieve the most common words
count.extend(temp.most_common(max_vocabulary_size))

'''
biglis = []
for t in count:
    biglis.append(t[0])
for l in listLabel:
    if l not in biglis:
        count.append((l, temp[l]))
'''
# Remove samples with less than 'min_occurrence' occurrences
i = len(count)-1;
while i >= 0:
    if count[i][1] < min_occurrence:
        #print(count[i])
        count.pop(i)
        i -= 1
    else:
        break # The collection is ordered, so stop when 'min_occurrence' is reached
    

# Compute the vocabulary size
vocabulary_size = len(count)
# Assign an id to each word
word2id = dict()
for i, (word, _)in enumerate(count):
    word2id[word] = i

data = list()
unk_c = 0
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
    index = word2id.get(word, 0)
    if index == 0:
        unk_c += 1
    data.append(index)
count[0] = ('<unk>', unk_c)
id2word = dict(zip(word2id.values(), word2id.keys()))

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])

data_index = 0
# Generate training batch for the skip-gram model
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# Input data
X = tf.placeholder(tf.int32, shape=[None])
# Input label
Y = tf.placeholder(tf.int32, shape=[None, 1])

# Ensure the following ops & var are assigned on CPU
# (some ops are not compatible on GPU)
with tf.device('/gpu:0'):
    # Create the embedding variable (each row represent a word embedding vector)
    embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size], stddev=0.1)) #shibo, default std is too large
    # Lookup the corresponding embedding vectors for each sample in X
    X_embed = tf.nn.embedding_lookup(embedding, X)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch
loss_op = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=Y,
                   inputs=X_embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))

# Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluation
# Compute the cosine similarity between input data embedding and every embedding vectors
X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Testing data
    x_test = np.array([word2id[w] for w in eval_words])

    average_loss = 0
    for step in range(1, num_steps + 1): ## change xrange to range, Shibo
        # Get a new batch of data
        batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
        # Run training op
        _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        average_loss += loss

        if step % display_step == 0 or step == 1:
            if step > 1:
                average_loss /= display_step
            print("Step " + str(step) + ", Average Loss= " + \
                  "{:.4f}".format(average_loss))
            average_loss = 0

        # Evaluation
        if step % eval_step == 0 or step == 1:
            print("Evaluation...")
            sim = sess.run(cosine_sim_op, feed_dict={X: x_test})
            for i in range(len(eval_words)): ## change xrange to range
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(top_k):  ## Shibo
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)

        if step == num_steps:
            embeddings = embedding.eval() # save embedding result
            wordList = list(id2word.values())
            semantic = pd.DataFrame(embeddings, index=wordList)
            semantic.to_csv('semantic_'+str(min_occurrence)+'.txt', sep=' ', index=True, header=False)
            print("Embeddings saved.")
