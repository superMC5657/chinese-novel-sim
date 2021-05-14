# -*- coding: utf-8 -*-
# !@time: 2021/5/14 上午12:53
# !@author: superMC @email: 18758266469@163.com
# !@fileName: idf.py

import os
import joblib

from algorithm.utils import mkdirs
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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
