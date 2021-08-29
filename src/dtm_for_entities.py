import re
from nltk.corpus import stopwords
from gensim.models import LdaSeqModel
import os
import unicodedata
from bs4 import BeautifulSoup
import datetime

import dateutil.parser

from functools import reduce
import operator

import json
import en_core_web_sm
import numpy as np
import string
from gensim import corpora
import pickle


nlp = en_core_web_sm.load()
stop_words = stopwords.words('english')  + list(string.punctuation)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'^[0-9]+\.', '', text)

    w = nlp(text)
    words = [w[i].lemma_ for i in range(len(w))]
    words = [s for s in words if s not in stop_words and s != "-PRON-" and s != "-" and s != "—" and not str.isspace(s)]
    return words

def get_text(xml_file_src):

    with open(xml_file_src, 'r') as f:
        data = f.read()

    Bs_data= BeautifulSoup(data, 'xml')

    body = Bs_data.find_all('p')
    paragraphs = [p.text for p in body]
    if len(paragraphs) == 0:
        return False, None, None
    if len(Bs_data.find_all('hedline')) == 0:
        headline = 'No Headline'
    else:
        headline = Bs_data.find_all('hedline')[0].text
    article = reduce(operator.concat, paragraphs)  #article is a string

    return True,article, headline


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


def get_BOW(text_data):
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    dummy = dictionary.get(0)
    id2token = dictionary.id2token

    return corpus, id2token, dictionary


def get_corpus(source_news):
    text_data = reduce(operator.concat, [x[1] for x in source_news])
    corpus, id2token, dictionary = get_BOW(text_data)
    return corpus, id2token, dictionary


def get_nyt_articles(src_dir, years,entities, save_dir):
    # columns: week, text
    monthly_articles = {}
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for year in years:

        year_src = os.path.join(src_dir, year)
        for month in months:

            month_src = os.path.join(year_src, month)
            print(month_src)
            days = os.listdir(month_src)
            if ".DS_Store" in days:
                days.remove(".DS_Store")
            month_articles = []
            for day in days:
                day_articles = []
                day_src = os.path.join(month_src, day)
                print(day_src)
                articles = os.listdir(day_src)
                if ".DS_Store" in articles:
                    articles.remove(".DS_Store")
                for article in articles:

                    article_src = os.path.join(day_src, article)
                    has, text, headline = get_text(article_src)
                    if not has:
                        continue

                    words = clean_text(text)
                    found = False
                    for entity in entities:
                        if find_existence(words, entity)[0]:
                            found = True
                            print(entity, 'found')
                    if not found:
                        continue
                    day_articles.append((words,text, headline ,int(day)))
                month_articles += day_articles


            print(os.path.join(save_dir, year + '_' + month))
            pickle.dump(month_articles,open(os.path.join(save_dir, 'monthly',year + '_' + month + '.pkl'), 'wb'))
            monthly_articles[year + '-' + month] = [article[0] for article in month_articles]

    monthly_articles = sorted(monthly_articles.items(), key=lambda x: x[0],
                          reverse=False)  # sorting by ascending order of year_month_articles
    return monthly_articles

def get_articles(src_dirs, entities):
    # columns: week, text

    week_to_text = {}
    start_time = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    start_time_week_day = start_time.weekday()
    count = 0
    for src_dir in src_dirs:
        with open(src_dir, mode="r") as reader:
            f = json.load(reader)
            for article in f:
                count += 1
                text = unicodedata.normalize('NFKD', article["body"]).replace("\n", " ").replace(r'\w+', "")
                words = clean_text(text)
                found = False
                for entity in entities:
                    if find_existence(words, entity)[0]:
                        found = True

                if not found:
                    continue
                pubtime = dateutil.parser.parse(article["published_at"])  # Python 3.6
                days_from_start = (pubtime - start_time).days  # int
                week_num = int((days_from_start + start_time_week_day) / 7)

                if week_num in week_to_text:
                    week_to_text[week_num].append(words)
                else:
                    week_to_text[week_num] = [words]
                print(count)

    week_to_text = sorted(week_to_text.items(), key=lambda x: x[0], reverse=False)  # sorting by asceding order of weeks

    return week_to_text


def main(entities, data_source, source_dir, save_path, num_topics, year1, year2):

    '''

    :param entities: all forms of the same entity. e.g.: ['Bill Clinton', 'bill clinton', 'president clinton']
    :param data_source: 'NYT' or 'COVID_news'
    :param source_dir: e.g.: 'data/NYT' or 'data/covid_news'/
    :param num_topics: 10
    :param year1: if 'NYT', provide two years for a period of time. e.g.,: 1997
    :param year2: e.g., 1998
    :return:
    '''

    if data_source == 'COVID_news':

        list_of_sources = os.listdir(source_dir)
        src_dirs = [source_dir + source for source in list_of_sources]
        bow_articles = get_articles(src_dirs = src_dirs, entities = entities)


    elif data_source == 'NYT':

        years = os.listdir(source_dir)
        years = list(map(int, years))
        filtered_years = [str(y) for y in years if y >= year1 and y <= year2]
        bow_articles = get_nyt_articles(src_dir=source_dir, years=filtered_years, entities=entities,
                                            save_dir=save_path)

    dtm_corpus, dtm_id2token, dtm_dictionary = get_corpus(bow_articles)
    article_count = [len(x[1]) for x in bow_articles]
    count = [x[0] for x in bow_articles]

    pickle.dump(dtm_dictionary, open(save_path + "/dictionary.pkl", 'wb'))

    pickle.dump(dtm_corpus, open(save_path + "/dtm_corpus.pkl", 'wb'))
    pickle.dump(dtm_id2token, open(save_path + "/id2token.pkl", 'wb'))
    pickle.dump(article_count, open(save_path + "/article_count.pkl", 'wb'))
    pickle.dump(count, open(save_path + "/count.pkl", 'wb'))

    ldaseq = LdaSeqModel(corpus=dtm_corpus, id2word=dtm_id2token,
                         time_slice=article_count, num_topics=num_topics, chunksize=1)

    pickle.dump(ldaseq, open(save_path + "/ldaseq.pkl", 'wb'))












