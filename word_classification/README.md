#1. Introduction
Domain Word Embedding via Matrix Local Reconstruction, which helps transfer reliable knowledge from the Domain Statistic Space to the Semantic Space. 


#2. Requirements
python 3.X

for network requests, grabing html and parsing contents:
bs4, lxml, urllib

for extracting articles:
newspaper3k

for preprocessing text and tokenization:
C compiler

for training:
Tensorflow 


#3. Usage
##Data Gathering 
"crawl" folder. 

Can skip data gathering and start from preprocessing. 
"python extract_500urls.py"
"python extract_children.py delay", e.g. "python extract_children.py 1"
"python url2article.py"

##Processing
"process" folder. 

"./preprocess raw.txt"

The preprocessing contains several parts: remove [edit] and [reference] from wiki articles, remove punctuations, remove words containing non-ASCII characters, convert all upper cases to lower, tokenize n-gram, remove stopwords, remove extra whitespaces and save as "processed.txt". 

If you want to customize the punctuation list, go to "nlt.h" and add/remove punctuations. Recompile the C code, "gcc -o preprocess preprocess.c". 

You may adapt the n-gram list, which will later be converted to tokens, by making changes to the .csv file. Notice that this is closely related to the word embedding domain. 

If you want to customize the stopword list, go to "stopwords.txt" and add/remove words. Note that a single "," is used as delimiter. 

##Training Word2Vec
"word2vec" folder. 

After preprocessing, you can train the word2vec by "python word2vec.py". 

There are two options here, discard of the low-frequency domain words and train word2vec or include them manually before training. The outputs will be marked with "full" and "part". 

It is strongly recommended to use GPU for training. It takes several hours to get satisfying result on NVIDIA GP100. 

If you desire to save terminal output to a file, run as "python word2vec.py |& tee log.txt" so that you can compare the result given different model settings. Refer to
https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file/420983

##Get Domain Affinity Matrix and Word Embedding Matrix
"matrices" folder. 

The word embeddings given by the above step might be not reliable, especially one is working on domain-specific language tasks, due to words' low frequency, e.g. disease and chemical molecules. To overcome this, one may leverage the knowledge that is well-formed in existing datasets. 

In here the affinity matrix being used is the correlation matrix based on stock historical time series. To get the affinity matrix, run "python domainMat.py". 

To get the word embedding matrix, run "python semanticMat.py full" or "python semantic.py part". 

##Matrix Local Reconstruction
"mlr" folder. 

run "python reconstruct.py"

The weight matrix and the reconstructed word embedding matrix will be saved. 

##Evaluation
"eval" folder. 

The classifier is merely a K-Nereast-Neighborhood. 

run "python semanticKNN.py" to see the result on original domain word embedding matrix. 

run "python KNN.py aff" to see the result on domain affinity matrix. 

run "python KNN.py PQ" to see the result reconstructed domain word embedding matrix. 



Feel free to apply this framework on other domains and datasets. 
espoyao (at) gmail [dot] com
