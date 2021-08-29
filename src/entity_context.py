import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import en_core_web_sm


V_DIM = 300
stop = stopwords.words('english') + list(string.punctuation)
nlp = en_core_web_sm.load()
categories = ["care.virtue", "care.vice", "fairness.virtue", "fairness.vice",
                 "loyalty.virtue", "loyalty.vice", "authority.virtue", "authority.vice",
                 "sanctity.virtue", "sanctity.vice"]


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



