import en_core_web_sm
import os
def preprocess_mfd2_lexicons(tokenize=False):
    mfd2_lexicon_categories = [
        "care.virtue", "care.vice",
        "fairness.virtue", "fairness.vice",
        "loyalty.virtue", "loyalty.vice",
        "authority.virtue", "authority.vice",
        "sanctity.virtue", "sanctity.vice"
    ]
    mfd2_lexicons = {}
    with open("../../data/mfd2.0.dic", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("%"):
            continue
        L = line.split()
        word = L[0]
        if len(word) < 3:
            continue
        cat_num = int(L[-1])
        cat = mfd2_lexicon_categories[cat_num-1]
        if cat not in mfd2_lexicons:
            mfd2_lexicons[cat] = [word]
        else:
            mfd2_lexicons[cat].append(word)

    if tokenize:

        nlp = en_core_web_sm.load()
        lexicons = {}
        for k in mfd2_lexicons:
            res = set()
            for w in mfd2_lexicons[k]:
                lem = nlp(w)[0].lemma_
                res.add(lem)
            lexicons[k] = list(res)
        mfd2_lexicons = lexicons

    return mfd2_lexicons
mfd2_lexicons = preprocess_mfd2_lexicons(True)
