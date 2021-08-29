import numpy as np
import pandas as pd
import en_core_web_md



class CoherenceFinder:
    def fit(self,df, feature):
        '''
        Args:
            df: a df of articles with columns headline, and body
            feature: headline of body used for coherence
        '''
        self.data = df
        self.doc_feature = feature
        self.nlp = en_core_web_md.load()



    def get_article_headlines(self):
        article_headlines = self.data['headline']
        article_headlines = [str(v).split('\n')[1] if len(str(v).split('\n')) > 1 else str(v) for v in article_headlines]
        return article_headlines


    def get_pairwise_similarity(self):
        '''
        :return coherence of the documents in the data
        '''

        if self.doc_feature == 'headline':
            docs = self.get_article_headlines()
        else:
            docs = self.data[self.doc_feature]


        doc_similarities = []

        doc_nlps = [self.nlp(doc) for doc in docs]

        for doc1 in doc_nlps:
            doc1_similarities = []
            for doc2 in doc_nlps:
                if doc1 != doc2:
                    similarity  = doc1.similarity(doc2)
                    doc1_similarities.append(similarity)

            doc_similarities.append(np.mean(doc1_similarities))

        pair_wise_similarity = np.mean(doc_similarities)
        return pair_wise_similarity



def get_topic_coherence_dist(df, step_col):
    '''
    Finds coherence for all the documents sets in df
        :param df: a dataframe containing subsets of documents
        :param step_col (str): column that separates subsets

        Returns: a tuple of
        - the input df with a new column coherence
        - coherence values of all the subsets
    '''
    steps = df[step_col].unique()
    all_dfs = pd.DataFrame()
    coherences = []
    step_num = min(10, len(steps))
    for step in range(step_num):
        new_df = df[df[step_col] == step]
        coherence_finder = CoherenceFinder()
        coherence_finder.fit(new_df, 'headline')
        coherence = coherence_finder.get_pairwise_similarity()
        new_df['coherence'] = [coherence] * len(new_df)
        coherences.append(coherence)
        all_dfs = all_dfs.append(new_df)


    return all_dfs, coherences

