# -*- coding: utf-8 -*-
# !@time: 2021/5/14 上午12:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: idf.py

import os
import joblib

from algorithm.utils import mkdirs
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from scipy.linalg import norm
from algorithm.utils import read_novel, read_stopwords, line_process


# IDF模型，保存模型之后可以重复利用
class IDF:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def save(self, model_dir):
        mkdirs(model_dir)
        cv_model_path = os.path.join(model_dir, 'cv.model')
        idf_model_path = os.path.join(model_dir, 'idf.model')

        joblib.dump(self.vectorizer, cv_model_path)
        joblib.dump(self.transformer, idf_model_path)

    def restore(self, model_dir):
        mkdirs(model_dir)
        cv_model_path = os.path.join(model_dir, 'cv.model')
        idf_model_path = os.path.join(model_dir, 'idf.model')

        self.vectorizer = joblib.load(cv_model_path)
        self.transformer = joblib.load(idf_model_path)

    def train(self, corpus):
        self.vectorizer.fit(corpus)
        x = self.vectorizer.transform(corpus)
        self.transformer.fit(x)

    def forward(self, corpus):
        x = self.vectorizer.transform(corpus)
        words = self.vectorizer.get_feature_names()
        tfidf = self.transformer.transform(x)
        weight = tfidf.toarray()
        return weight

    def __call__(self, corpus):
        return self.forward(corpus)




# 训练
def trainer():
    datasets_path = 'data/datasets'
    corpus = read_novel(datasets_path, stopwords)
    idf = IDF()
    idf.train(corpus)
    idf.save(checkpoint_path)

# 推理
def inference(s1, s2):
    corpus = []
    cor = line_process(s1, stopwords)
    corpus.append(cor)
    cor = line_process(s2, stopwords)
    corpus.append(cor)
    idf = IDF()
    idf.restore(checkpoint_path)
    try:
        vectors = idf(corpus)
    except:
        return False
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


if __name__ == '__main__':
    stopwords_path = 'data/StopWords.txt'
    checkpoint_path = 'algorithm/checkpoints/'
    stopwords = read_stopwords(stopwords_path)
    # trainer()
    s1 = '莲步微移，名为萧薰儿的少女行到魔石碑之前，小手伸出，镶着黑金丝的紫袖滑落而下，露出一截雪白娇嫩的皓腕，然后轻触着石碑。'
    s2 = '这名紫裙少女，论起美貌与气质来，比先前的萧媚，无疑还要更胜上几分，也难怪在场的众人都是这般动作。'
    sim = inference(s1, s2)
    print(sim)