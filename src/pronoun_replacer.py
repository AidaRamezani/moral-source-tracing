import neuralcoref
import en_core_web_sm

class PronounReplace:
    nlp = en_core_web_sm.load()
    neuralcoref.add_to_pipe(nlp)
    def replace(self, article):
        doc = self.nlp(article)
        reference_article = [str(token._.coref_clusters[-1].mentions[-1]._.coref_cluster.main ) if token._.in_coref else str(token) for token in doc]
        cleaned_referenced_article = []
        for i,v in enumerate(reference_article):
            if i > 0:
                if cleaned_referenced_article[-1] != reference_article[i]:
                    cleaned_referenced_article.append(v)
            else:
                cleaned_referenced_article.append(v)
        cleaned_referenced_article = " ".join(cleaned_referenced_article)
        return cleaned_referenced_article


