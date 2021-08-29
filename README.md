Readme

An unsupervised framework for tracing textual sources of moral change 


This repository contains replication code for the paper: Ramezani, A., Zhu, Z., Rudzicz, F., and Xu, Y. An unsupervised framework for tracing textual sources of moral change. Findings of the Association for Computational Linguistics: EMNLP 2021 (to appear).
=====================================================================


## Dependencies


Requires `python>=3.7`.

```python
numpy
pandas
pickle
statsmodels
changepoint
gensim
bs4
unicodedata
dateutil
json
nltk
spaCy
sklearn
neuralcoref
```

## Data
Moral Foundations Twitter Corpus: https://osf.io/k5n7y/

COVID-19 news source: https://aylien.com/blog/free-coronavirus-news-dataset


New York Times annotated corpus: https://catalog.ldc.upenn.edu/LDC2008T19


## Scripts

These scripts provide modules to run the main contributions and the
evaluation metrics in the paper.

-----

Change directories accordingly.
The usage of the different files are as follows:

`probabilistic_model.py` Implements the equations used in case study 1 based on the
probabilistic framework.


`dtm_for_entities.py` Implements and saves the Dynamic Topic Model for a given entity and its alternative forms.

`moral_sentiment.py` Saves the moral sentiments and topic probabilities for a given entity.

`change_point_mean_shift.py` Saves the change points in the moral sentiments time-series of a given entity.

`topic_based_model.py` Applies the topic-based model to a dataset. Given
a dataset of documents, moral sentiment dimension, a change point and a
time window, finds the most salient topic as the source of the moral
sentiment change.

`influence_function.py` Perturbs a dataset in a given window. Finds the
most influential set.

`random_perturbation.py` Perturbs a dataset in a given window by
removing documents randomly.

`coherence_finder.py` Applies the coherence evaluation on a set of
documents.


