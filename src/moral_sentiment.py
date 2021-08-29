from lexicons import mfd2_lexicons
from prepare_moral_models import clean_text, get_sentences, get_probability_prediction, \
    V_DIM, \
    emb, get_wv, get_moral_polarity_model, get_relevance_model, get_moral_sentiment_model,\
    find_existence, get_context_v2, get_bag_of_words, sharpen_context_2
from moral_models import get_neutral_words
import numpy as np
import pandas as pd
import json
import datetime
import dateutil
import unicodedata
import pickle
from functools import reduce
import operator
import os
from pronoun_replacer import PronounReplace


num_topic = 10


def get_probs(sentences, entities, r_model, p_model, s_model_pos, s_model_neg):
    exists = False

    context = []

    for s in sentences:
        if len(s) > 0:  # TODO refine
            for entity in entities:
                existence, indices = find_existence(s, entity)
                if existence:
                    exists = True
                    new_context = get_context_v2(s, indices, len(get_bag_of_words(entity)))

                    new_context_vec = [get_wv(emb, v) for v in new_context]
                    new_context_vec = [v for v in new_context_vec if not (v == np.zeros(V_DIM)).all()]

                    sentence_score, indices = sharpen_context_2(new_context_vec, r_model)


                    new_context_sharpened = [new_context_vec[x] for x in indices]  #contains only relevant words
                    # print('len context: ', len(context), end=" ")
                    context += new_context_sharpened
                    break


    if len(context) == 0:
        relevance_p = 0
        relevance_p_mean = 0
        polarity_p = 0
        polarity_p_mean = 0

        sentiment_p_average = {c: 0 for c in mfd2_lexicons}
        sentiment_p_average_mean = sentiment_p_average

    else:
        context_mean = np.mean(context, axis = 0)
        relevance_p_mean = get_probability_prediction([context_mean], r_model)[0]['moral']


        relevance_p = get_probability_prediction(context, r_model)
        relevance_p = np.mean([x['moral'] for x in relevance_p])


        polarity_p = get_probability_prediction(context, p_model)
        polarity_p = np.mean([x['+'] for x in polarity_p])
        polarity_p_mean = get_probability_prediction([context_mean], p_model)[0]['+']

        if polarity_p >= 0.5:
            sentiment_p = get_probability_prediction(context, s_model_pos)
        else:
            sentiment_p = get_probability_prediction(context, s_model_neg)
        sentiment_p_average = {c: [] for c in mfd2_lexicons}
        for w in sentiment_p:  # w is a dictionary
            for s in w:
                s_result = w[s]
                sentiment_p_average[s] += [s_result]

        for c in sentiment_p_average:
            if len(sentiment_p_average[c]) > 0:
                sentiment_p_average[c] = np.mean(sentiment_p_average[c])
            else:
                sentiment_p_average[c] = 0


        if polarity_p_mean >= 0.5:
            sentiment_p = get_probability_prediction([context_mean], s_model_pos)

        else:
            sentiment_p = get_probability_prediction([context_mean], s_model_neg)

        sentiment_p_average_mean = {c + '_mean': [] for c in mfd2_lexicons}
        for w in sentiment_p:  # w is a dictionary

            for s in w:
                s_result = w[s]
                sentiment_p_average_mean[s + '_mean'] += [s_result]

        for c in sentiment_p_average_mean:
            if len(sentiment_p_average_mean[c]) > 0:
                sentiment_p_average_mean[c] = np.mean(sentiment_p_average_mean[c])
            else:
                sentiment_p_average_mean[c] = 0

    sentiment_p_average.update(sentiment_p_average_mean)
    return exists,relevance_p, relevance_p_mean, polarity_p, polarity_p_mean, sentiment_p_average

def get_exisits(sentences, dtm_entity):
    for s in sentences:
        existence, indices = find_existence(s, dtm_entity)
        if existence:
            return True
    return False




def get_article_sentiments(replacer, text,relevance_mean,polarity_mean,sentiment, doc2bows, dtm_model, topic_dist, topics):

    text = replacer.replace(text)
    sentences = get_sentences(text)
    sentences = [clean_text(s) for s in sentences]
    exists, relevance_p, relevance_p_mean, polarity_p, polarity_p_mean, sentiment_p, \
        = get_probs(sentences, entities, r_model, p_model, s_model_pos, s_model_neg)

    if exists:

        relevance_mean.append(relevance_p_mean)
        polarity_mean.append(polarity_p_mean)
        sentiment.append(sentiment_p)
        doc = reduce(operator.concat, sentences)
        to_bow = dictionary.doc2bow(doc)
        doc2bows.append(to_bow)

        ts = dtm_model[to_bow]
        t = np.argmax(ts)
        topic_dist.append(ts)
        topics += [t]

    return exists

def get_date(file_name):

    date = file_name.split('_')
    year = date[0]
    month = date[1]
    return year, month



def get_topic_sentiment_v2(entities, src_dir, dtm_model,
                           save_dir, data_source = 'COVID_news', pickle_files = None ):
    replacer = PronounReplace()

    days = []
    dates = []
    head_lines = []
    publication = []
    relevance_mean = []
    polarity_mean = []
    sentiment = []
    article_id = []
    words = []
    topics = []
    topic_dist = []
    count = 0
    doc2bows = []
    main_entity = entities[0]


    if data_source == 'COVID_news':
        start_time = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
        days_range = [np.inf, -np.inf]
        start_time_week_day = start_time.weekday()
        weekdays = []
        weekdays_int = []
        week_nums = []
        sources = os.listdir(src_dir)
        sources_src_dir = [src_dir + source for source in sources]
        print(sources_src_dir)
        for source_src_dir in sources_src_dir:
            with open(source_src_dir, mode="r") as reader:
                f = json.load(reader)

                for article in f:
                    pubtime = dateutil.parser.parse(article["published_at"])
                    pmlink = article['links']['permalink']
                    root_src = pmlink.replace("//", "/").split("/")[1]

                    days_from_start = (pubtime - start_time).days  # int
                    weekday = pubtime.strftime("%A")
                    weekday_int = pubtime.weekday()
                    week_num = int((days_from_start + start_time_week_day) / 7)

                    days_range[0] = min(days_range[0], days_from_start)
                    days_range[1] = max(days_range[1], days_from_start)
                    text = unicodedata.normalize('NFKD', article["body"]).replace("\n", " ")
                    exists = get_article_sentiments(replacer, text, relevance_mean, polarity_mean, sentiment,
                                                    doc2bows, dtm_model, topic_dist, topics)
                    if exists:
                        id = article['id']
                        article_id += [id]

                        publication += [root_src]
                        dates += [days_from_start]
                        words += [main_entity]
                        weekdays += [weekday]
                        weekdays_int += [weekday_int]
                        week_nums += [week_num]
                        count += 1
                        print(count)




    elif data_source == 'NYT':
        months = []
        years =[]

        for pfile in pickle_files:
            file_dir = os.path.join(src_dir, pfile)
            month_articles = pickle.load(open(file_dir, "rb"))  # month articles is a list of articles
            year, month = get_date(pfile)
            print(year, month)
            for article in month_articles:  # month article has 30 days
                article_day = article[3]
                article_headline = article[2]
                article_text = article[1]  # article is not cleaned

                exists = get_article_sentiments(replacer, article_text, relevance_mean, polarity_mean, sentiment,
                                           doc2bows, dtm_model, topic_dist, topics)
                if exists:
                    days += [article_day]
                    months += [month]
                    years += [year]
                    head_lines += [article_headline]
                    words += [main_entity]

                    count += 1

                    print(count)

    df = pd.DataFrame(sentiment)

    topic_dist = np.array(topic_dist)
    for topic in range(num_topic):
        df[("topic" + str(topic))] = topic_dist[:, topic]

    df["main_topic"] = topics
    df["word"] = words

    df['relevance_p_mean'] = relevance_mean

    df['polarity_p_mean'] = polarity_mean
    df["date"] = dates
    df["doc2bows"] = doc2bows

    if data_source == 'COVID_news':
        df["week"] = week_nums
        df["weekday"] = weekdays
        df["weekday_int"] = weekdays_int
        df["publication"] = publication
        df["article_id"] = article_id

    elif data_source == 'NYT':
        df["months"] = months
        df["years"] = years
        df["headline"] = head_lines

    df.to_csv(save_dir)

def get_monthly_pickles(src_dir):
    pickle_files = os.listdir(src_dir)
    return pickle_files

#Example to run:

moral_words = reduce(operator.concat, list(mfd2_lexicons.values()))
moral_center = np.mean([get_wv(emb, mv) for mv in moral_words], axis=0).reshape(1, -1)

n_words = get_neutral_words()
r_model = get_relevance_model(neutral_words=n_words)
p_model = get_moral_polarity_model()
s_model_pos, s_model_neg = get_moral_sentiment_model()

data_source = 'NYT'
src_dir = 'data/NYT/billclinton' #from New York Times annotated corpus, all articles including bill clinton
entities = ['bill clinton']
dtm_src = 'data/dtm/billclinton/' #path to dynamic topic model (generated by dtm_for_entities.py)
save_dir = 'data/moral_sentiments/NYT/billclinton.csv'

ldaseq_model = pickle.load(open(dtm_src + '/ldaseq.pkl', 'rb'))

dictionary = pickle.load(open(dtm_src + "/dictionary.pkl", 'rb'))
pickle_files = get_monthly_pickles(src_dir)

get_topic_sentiment_v2(entities, src_dir, ldaseq_model,save_dir, data_source, pickle_files=pickle_files)


