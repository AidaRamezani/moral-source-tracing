from sklearn.utils import shuffle
import numpy as np
import pandas as pd

third_tiers = ['care.virtue', 'care.vice', 'fairness.virtue',
       'fairness.vice', 'loyalty.virtue', 'loyalty.vice', 'authority.virtue',
       'authority.vice', 'sanctity.virtue', 'sanctity.vice']


class TopicBased:
    def __init__(self, df, B, k):
        '''

        :param df: a dataset of all the documents in the corpus for each entity. each row is a document, columns are p_e(m | d)s, p (o | d)s, and publication time
        :param B: permutation repeat size (10000)
        :param k: identifies size of influence set
        '''
        self.data = df
        self.B = B
        self.k = k




    def perturb(self, start_point, end_point, topics, time_feature):

        '''
        perturbs the dataset by removing M documents published in (start_point, end_point] with the highest p(topics | d)
        Args:
            start_point: change point
            end_point: end of window
            topics: a list of topics for perturbation
            time_feature: time scale in the dataset
        Returns: a tuple of
        - a df perturbed dataset
        - a df of all the documents in the original dataset from (start_point, end_point]
        - a df of all the documents in the original dataset from (0, start_point]
        - a df of all the documents in the original dataset from (end_point, )
        - a df of the documents removed in the perturbed dataset
        '''


        middle_df = self.data.loc[self.data[time_feature] > start_point].loc[self.data[time_feature] <= end_point]
        rest_df1 = self.data.loc[self.data[time_feature] <= start_point]
        rest_df2 = self.data.loc[self.data[time_feature] > end_point]
        m = int(self.k * len(middle_df))
        split_length = int(m / len(topics))
        keeping_df = middle_df.copy(deep = True)
        removing_dfs = pd.DataFrame()

        for topic in topics:
            selected_topics = np.array(keeping_df['topic' + str(topic)])
            keeping_df['selected_topics'] = selected_topics
            new_df_copy = keeping_df.copy(deep = True)
            new_df_copy = new_df_copy.sort_values(by=['selected_topics'])[::-1]
            removing_df = new_df_copy[:split_length]
            removing_dfs = removing_dfs.append(removing_df)

            keeping_df = new_df_copy[split_length:]

        keeping_new_df = keeping_df
        perturbed = rest_df1.append(rest_df2).append(keeping_new_df)
        return perturbed, middle_df, rest_df1, rest_df2, removing_dfs

    def random_perturb(self, middle_df, rest_df1, rest_df2 ):
        '''
        Randomly removes M documents from the middle_df

        '''

        m = int(self.k * len(middle_df))
        new_df2 = shuffle(middle_df)
        keeping_new_df = new_df2[m:]
        perturbed = rest_df1.append(rest_df2).append(keeping_new_df)
        return perturbed



    def get_j(self, df, m_sentiment, start_point, end_point, time_feature): #Delta{J}
        '''
        Finds dJ in df which is a dataset perturbed in (start_point, end_point]

        '''

        p_sentiment_t2_df = df.loc[df[time_feature] > start_point].loc[df[time_feature] <= end_point]
        p_sentiment_t2= np.array(p_sentiment_t2_df.groupby([time_feature]).mean()[m_sentiment])
        p_sentiment_t2_mean = np.mean(p_sentiment_t2)
        p_sentiment_t1_df = df.loc[df[time_feature] == start_point][m_sentiment]
        p_sentiment_t1 = np.mean(p_sentiment_t1_df)


        return abs(p_sentiment_t2_mean - p_sentiment_t1)




    def get_p_value(self, m_sentiment, start_point, end_point, topics,time_feature, B = None):

        '''
        Perturbs the original dataset based on a list topics
        Returns:
            - dJ of the perturbed dataset
            - p-value of dJ compared to a null random baseline
        '''
        perturbed, new_df, rest_df1, rest_df2 , _= self.perturb(start_point, end_point, topics,time_feature)


        j  = self.get_j(perturbed, m_sentiment, start_point, end_point, time_feature)

        if B == None:
            B = self.B
        permutation_js = []

        for i in range(B):
            df = self.random_perturb(new_df, rest_df1, rest_df2)
            new_j = self.get_j(df, m_sentiment, start_point, end_point, time_feature)

            permutation_js.append(new_j)


        p_value = np.sum(np.array(permutation_js) < j) / B

        return j ,p_value


    def get_p_sentiment_given_topic_bar(self, df, m_sentiment, topics, tier = 1):
        '''
        Returns:
            p(m |e, dT, topic != topics)
        '''
        if tier != 3:
            p_sentiment_pos_for_docs = np.array(df[m_sentiment])
            p_sentiment_neg_for_docs = np.array(1 - df[m_sentiment])

            selected_topics = np.zeros(len(df))
            for topic in topics:
                selected_topics += np.array(df['topic' + str(topic)])

            p_topic_for_docs = 1 - selected_topics


            p_sentiment_p = np.sum(p_sentiment_pos_for_docs * p_topic_for_docs)
            p_sentiment_n = np.sum(p_sentiment_neg_for_docs * p_topic_for_docs)

            return p_sentiment_p / (p_sentiment_p + p_sentiment_n)

        else:
            p_sentiment_for_docs = np.array(df[m_sentiment])
            p_sentiment_all_for_docs = [np.array(df[s + '_mean']) for s in third_tiers]

            selected_topics = np.zeros(len(df))
            for topic in topics:
                selected_topics += np.array(df['topic' + str(topic)])

            p_topic_for_docs = 1 - selected_topics

            p_sentiment_p = np.sum(p_sentiment_for_docs * p_topic_for_docs)
            p_sentiment_all = np.sum([np.sum(p_sentiment * p_topic_for_docs) for p_sentiment in p_sentiment_all_for_docs])

            return p_sentiment_p / p_sentiment_all



    def topic_moral_sentiment_change(self, m_sentiment, start_point, end_point, topics, time_feature, tier): #Delta{S}
        '''
        Returns:
            dS(e, m , topics, start_point, (start_point, end_point])
        '''
        new_df = self.data.loc[self.data[time_feature] > start_point].loc[self.data[time_feature] <= end_point]
        p_sentiment_given_topic_bar = self.get_p_sentiment_given_topic_bar(new_df, m_sentiment, topics, tier)
        p_sentiment_t1_df = self.data.loc[self.data[time_feature] == start_point][m_sentiment]
        p_sentiment_t1 = np.mean(p_sentiment_t1_df)

        return abs(p_sentiment_given_topic_bar - p_sentiment_t1)


    def get_salient_topic(self, moral_dimension, tier, start_point, end_point, time_feature, topics):
        '''
        Args:
            moral_dimension: moral sentiment dimension
            tier: 1, 2 ,3
            start_point: change point
            end_point: end of window
            time_feature: time scale in the data
            topics: a list of topics

        Returns: a tuple of
            - the most salient topic (source)
            - most salient topic's dS
            - most salient topic's dJ
            - most salient topic's p-value of dJ significance
        '''
        topics_change = [(t , self.topic_moral_sentiment_change(moral_dimension, start_point, end_point, [t], time_feature,
                                                             tier),
                         self.get_p_value(
                             moral_dimension, start_point, end_point, [t], time_feature, tier))

                         for t in topics
                         ]

        #sorting topics based on smallest dS
        topics_change.sort()
        return topics_change[0]





