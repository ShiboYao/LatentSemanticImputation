import string
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer

def removePunc(s):
    punc = string.punctuation
    translater = str.maketrans(punc, ' '*len(punc))
    s = s.translate(translater)

    return s


def removeNonASCII(s):
    s = ''.join([c if ord(c) < 128 else ' ' for c in s])

    return s


def removeStop(s, stop_words = None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    l = s.split()
    l = [w for w in l if not w in stop_words]
    s = ' '.join(l)

    return s


def tokenSentence(s):
    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(s)
    l = tokenizer.tokenize(s)
    s = '\n'.join(l)

    return s


