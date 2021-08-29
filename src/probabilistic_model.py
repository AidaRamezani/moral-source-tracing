import numpy as np

third_tiers = ['care.virtue', 'care.vice', 'fairness.virtue',
       'fairness.vice', 'loyalty.virtue', 'loyalty.vice', 'authority.virtue',
       'authority.vice', 'sanctity.virtue', 'sanctity.vice']



def get_sentiment_entity(df, entity, m_sentiment, tier):
    '''
    :return: p_emp(m | e)
    '''
    new_df = df.loc[df["word"] == entity]
    if tier == 3:
        new_df = new_df.loc[new_df["relevance_label"] > 0]
        if m_sentiment.endswith("vice"):
            new_df = new_df.loc[new_df["polarity_label"] < 0]
        else:
            new_df = new_df.loc[new_df["polarity_label"] > 0]


        n = len(new_df)
        if n == 0:
            return None, None
        array = []
        p_sentiment = 0
        for cat in third_tiers:
            article_sentiment = len(new_df.loc[new_df["category_label"] == cat]) / n
            array += [article_sentiment]
            if cat == m_sentiment:
                p_sentiment = article_sentiment

        p_sentiment /= np.sum(array)
        array = np.array(array) / np.sum(array)
        res = p_sentiment, dict(zip(third_tiers, array))
        return res


    elif tier == 2:
        new_df = new_df.loc[new_df["relevance_label"] == 1]
        n = len(new_df)
        if n == 0:
            return None, None
        array = np.array([len(new_df.loc[new_df["polarity_label"] > 0]) / n, len(new_df.loc[new_df["polarity_label"] < 0]) / n])
        array /= np.sum(array)
        return array


    else:
        n = len(new_df)
        if n == 0:
            return None, None
        p = len(new_df.loc[new_df["relevance_label"] > 0]) / n
        return p, 1 - p


def get_sentiment_topic_entity(df, entity, m_sentiment, topic, tier):
    '''

    :return: p_emp(m | e , o)
    '''

    new_df = df.loc[df["word"] == entity].loc[df["topic"] == topic]
    return get_sentiment_entity(new_df, entity, m_sentiment, tier)



def get_sentiment_entity_prob(df, entity, m_sentiment, tier):
    '''

    :return: p(m | e)
    '''
    new_df = df.loc[df["word"] == entity]

    if tier == 3:
        new_df = new_df.loc[new_df["log_odds_relevance"] > 0]
        if m_sentiment.endswith("vice"):
            new_df = new_df.loc[new_df["log_odds_polarity"] < 0]

        else:
            new_df = new_df.loc[new_df["log_odds_polarity"] > 0]


        n = len(new_df)

        if n == 0:
            return None, None

        array = []
        p_sentiment = 0

        for cat in third_tiers:
            article_sentiment = np.sum(new_df[cat] / n)
            array += [article_sentiment]
            if cat == m_sentiment:
                p_sentiment = article_sentiment

        p_sentiment /= np.sum(array)
        array = np.array(array) / np.sum(array)
        res = p_sentiment, dict(zip(third_tiers, array))



    else:
        if tier == 2:
            new_df = new_df.loc[new_df["log_odds_relevance"] > 0]

        n = len(new_df)
        if n == 0:
            return None, None

        array = np.array([np.sum(new_df[m_sentiment] / n), np.sum((1 - new_df[m_sentiment]) / n)])
        array = array / np.sum(array)
        res = array

    return res


def get_sentiment_topic_entity_prob_simple(df, entity, m_sentiment, topic, tier):
    '''
    :return p (m | e , o)


        p(tweet | entity , topic ) = p(topic | entity, tweet) p (tweet | entity) / p(topic | entity)
        p (topic | entity , tweet) = p (topic | tweet) 0 or 1 (from corpus)
        p( tweet | entity) = 1 / n
        p (topic | entity) = 1 / k

    '''


    new_df = df.loc[df["word"] == entity]
    if tier == 3:
        new_df = new_df.loc[new_df["log_odds_relevance"] > 0]

        if m_sentiment.endswith("vice"):
            new_df = new_df.loc[new_df["log_odds_polarity"] < 0]

        else:
            new_df = new_df.loc[new_df["log_odds_polarity"] > 0]

        n = len(new_df)
        k = len(new_df.loc[new_df["topic"] == topic])

        if k == 0 or n == 0:
            return None, None
        array = []

        p_topic_tweet = np.array((new_df['topic'] == topic) * 1)

        p_sentiment = 0
        for cat in third_tiers:

            article_sentiment = np.sum((new_df[cat] * p_topic_tweet) / k)
            array += [article_sentiment]
            if cat == m_sentiment:
                p_sentiment = article_sentiment

        p_sentiment /= np.sum(array)
        array = np.array(array) / np.sum(array)
        res = p_sentiment, dict(zip(third_tiers, array))


    else:
        if tier == 2:
            new_df = new_df.loc[new_df["log_odds_relevance"] > 0]
        n = len(new_df)
        k = len(new_df.loc[new_df["topic"] == topic])
        if k == 0 or n == 0:
            return None, None
        p_topic_tweet = np.array((new_df['topic'] == topic) * 1)

        pos_sum = np.sum((new_df[m_sentiment] * p_topic_tweet) / k)
        neg_sum = np.sum(((1 - new_df[m_sentiment]) * p_topic_tweet) / k)
        array = [pos_sum, neg_sum]
        array = np.array(array) / np.sum(array)
        res = array

    return res
