import numpy as np
import pandas as pd
from lexicons import mfd2_lexicons
from functools import reduce
import operator
import seaborn as sns; sns.set(color_codes=True)
from moral_models import CentroidModel

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import en_core_web_sm
from gensim import models

V_DIM = 300
stop = stopwords.words('english') + list(string.punctuation)

nlp = en_core_web_sm.load()
categories = ["care.virtue", "care.vice", "fairness.virtue", "fairness.vice",
                 "loyalty.virtue", "loyalty.vice", "authority.virtue", "authority.vice",
                 "sanctity.virtue", "sanctity.vice"]



def read_emb():
    w = models.KeyedVectors.load_word2vec_format(
        '../../data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    return w
emb = read_emb()
def get_wv(emb, word):

    if word in emb.vocab:

        return emb[word]

    else:

        OOV = np.zeros(V_DIM)
        return OOV


def get_bag_of_words(text, min_freq=1):
    D = {}
    for w in text.split():
        if w not in D:
            D[w] = 1
        else:
            D[w] += 1
    bag = list(filter(lambda w: D[w] >= min_freq, D.keys()))
    return bag


def find_existence(text, w):



    bag_of_words_w = get_bag_of_words(w)

    final_indices = []
    if len(bag_of_words_w) < 1:
        return False, None
    first_token = bag_of_words_w[0]
    if first_token in text:
        token_indices =  list(np.where(np.array(text) == first_token)[0])


        for ind in token_indices:
            correct_ind = True
            for j in range(1, len(bag_of_words_w)):
                if len(text) <= (ind + j) or bag_of_words_w[j] != text[ind + j]:
                    correct_ind = False
            if correct_ind:
                final_indices.append(ind)
    else:
        return False, None

    if len(final_indices) >0:


        return True, final_indices
    else:
        return False, None


def get_context_v2(text, indices ,entity_len): #returns the whole sentence as the contex
    new_text = []
    previous = 0
    for i in range(len(indices)):
        index = indices[i]
        new_text += text[previous: index]
        previous += index + entity_len
    new_text += text[previous:]
    return new_text


def clean_text(sentence):
    sentence = sentence.lower()
    w = nlp(sentence)
    sentence = [w[i].lemma_ for i in range(len(w))]
    sentence = [s for s in sentence if s not in stop and s != "-PRON-" and s != "-"]
    return sentence

def get_sentences(text):
    return sent_tokenize(text)

def sharpen_context_2(context,r_model):


    prediction = r_model.predict(context)

    score = np.sum(np.array(prediction) == 'moral') / len(context)
    indices = [i for i in range(len(context)) if prediction[i] == 'moral']



    return score, indices

def get_probability_prediction(data, relevance_model):
    return relevance_model.predict_proba(data)


def get_relevance_model(neutral_words ,lexicon = mfd2_lexicons):
    centroid_model = CentroidModel()
    moral_words =  reduce(operator.concat, list(lexicon.values()))
    all_words = neutral_words + moral_words
    all_words_vec = [get_wv(emb, v) for v in all_words]
    labels = ["neutral" for n in neutral_words]
    labels += ["moral" for m in moral_words]
    df = pd.DataFrame(all_words_vec)
    df["label"] = labels
    centroid_model.fit(df, label= "label", feature_cols= [i for i in range(V_DIM)])

    return centroid_model



def get_moral_polarity_model(lexicon = mfd2_lexicons):
    centroid_model = CentroidModel()
    # words = reduce(operator.concat, list(lexicon.values()))
    words = []
    labels = []
    for i in categories:
        words += lexicon[i]
        if i.endswith("virtue"):
            labels += ['+'] * len(lexicon[i])
        else:
            labels += ['-'] * len(lexicon[i])


    words_vec = [get_wv(emb, v) for v in words]


    df = pd.DataFrame(words_vec)
    df["label"] = labels
    centroid_model.fit(df, label="label", feature_cols=[i for i in range(V_DIM)])

    return centroid_model






def get_moral_sentiment_model(lexicon=mfd2_lexicons):
    centroid_model_pos = CentroidModel()
    centroid_model_neg = CentroidModel()
    words_pos = []
    labels_pos = []
    categories = list(lexicon.keys())
    words_neg = []
    labels_neg = []

    for i in categories:
        if i.endswith('virtue'):
            words_pos += lexicon[i]
            labels_pos += [i] * len(lexicon[i])
        else:
            words_neg += lexicon[i]
            labels_neg += [i] * len(lexicon[i])

    # words = reduce(operator.concat, list(lexicon.values()))
    words_vec_pos = [get_wv(emb, v) for v in words_pos]
    words_vec_neg = [get_wv(emb, v) for v in words_neg]

    df_pos = pd.DataFrame(words_vec_pos)
    df_pos["label"] = labels_pos
    centroid_model_pos.fit(df_pos, label="label", feature_cols=[i for i in range(V_DIM)])

    df_neg = pd.DataFrame(words_vec_neg)
    df_neg["label"] = labels_neg
    centroid_model_neg.fit(df_neg, label="label", feature_cols=[i for i in range(V_DIM)])

    return centroid_model_pos, centroid_model_neg

















