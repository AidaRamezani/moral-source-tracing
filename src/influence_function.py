from sklearn.utils import shuffle
import numpy as np


class InfluenceFinder:
    def __init__(self, df, B, k):
        '''

        :param df: a dataset of all the documents in the corpus for each entity. each row is a document, columns are p_e(m | d)s,  p (o | d)s and publication times
        :param B: permutation repeat size (10000)
        :param k: identifies size of influence set
        '''
        self.data = df
        self.B = B
        self.k = k

    def get_df(self, start_point, end_point, time_feature):
        middle_df = self.data.loc[self.data[time_feature] > start_point].loc[self.data[time_feature] <= end_point]
        rest_df1 = self.data.loc[self.data[time_feature] <= start_point]
        rest_df2 = self.data.loc[self.data[time_feature] > end_point]
        return middle_df, rest_df1, rest_df2

    def find_influence(self,start_point, end_point,m_sentiment, time_feature):
        '''

        Args:
            start_point: changing point
            end_point: end of window
            m_sentiment: moral sentiment dimension 'moral relevance', 'moral polarity', or 10 fine-grained foundation categories in MFT
            time_feature: time scale


        Returns: a tuple with
            - the influence set
            - Delta_{J} of influence set
            - Delta{J}s of null baseline
        '''

        middle_df, rest_df1, rest_df2 = self.get_df(start_point, end_point, time_feature)
        js = []

        min_j = None

        influence_df = None
        for b in range(self.B):
            perturbed_df, keeping_new_df, removing_df = self.random_perturb(middle_df, rest_df1 ,rest_df2)
            new_j = self.get_j(perturbed_df, m_sentiment, start_point, end_point, time_feature)

            js.append(new_j)

            if min_j == None or new_j < min_j:
                min_j = new_j
                influence_df = removing_df



        influence_df['j'] = [min_j] * len(influence_df)
        return (influence_df, min_j, js)


    def random_perturb(self, middle_df, rest_df1, rest_df2):
        '''
        Args:
            middle_df: a set of documents in Delta{t}
            rest_df1: documents before t
            rest_df2: documents after t + Delta{t}
            k: identifies the influence set size
        Returns: a tuple
            - the new perturbed dataset
            - documents in the changing window that are not removed
            - documents removed from the changing window in perturbation

        '''

        if self.k == None:
            m = 1
        else:
            m = int(self.k * len(middle_df))

        new_df = shuffle(middle_df)
        keeping_new_df = new_df[m:]
        removing_df = new_df[:m]
        perturbed = rest_df1.append(rest_df2).append(keeping_new_df)
        return (perturbed, keeping_new_df, removing_df)



    def get_pvalue(self, start_point, end_point,m_sentiment, time_feature):
        '''
        Args:
            :param start_point: changing point
            :param end_point: end of window
            :param m_sentiment: moral sentiment dimension
            :param time_feature: time scale
        Returns: a tuple of
            - influence set
            - Delta_{J} of influence set
            - degree of significance of the influence set Delta{J}

        '''
        influence_df, min_j, js = self.find_influence(start_point, end_point,m_sentiment, time_feature)
        p_value = np.sum(np.array(js) < min_j) / len(js)

        return (influence_df, min_j, p_value)


    def get_j(self, df, m_sentiment, start_point, end_point, time_feature):
        '''
        Args:
            :param df: a dataset of documents, columns are p(m | d)s and p (o | d)s.
            :param m_sentiment: moral sentiment dimension
            :param start_week: changing point
            :param end_week: end of window
            :param time_feature: time scale



        :return Delta{J} of the documents in df in the given window


        '''
        p_sentiment_t2_df = df.loc[df[time_feature] > start_point].loc[df[time_feature] <= end_point]
        p_sentiment_t2= np.array(p_sentiment_t2_df.groupby([time_feature]).mean()[m_sentiment])
        p_sentiment_t2_mean = np.mean(p_sentiment_t2)
        p_sentiment_t1_df = df.loc[df[time_feature] == start_point][m_sentiment]
        p_sentiment_t1 = np.mean(p_sentiment_t1_df)

        return abs(p_sentiment_t2_mean - p_sentiment_t1)

