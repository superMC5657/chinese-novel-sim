# -*- coding: utf-8 -*-
# !@time: 2021/5/15 上午12:01
# !@author: superMC @email: 18758266469@163.com
# !@fileName: web_flask.py
import os

import cv2
import hanlp
import numpy as np
from ddparser import DDParser
from flask import Flask, render_template, request
from scipy.linalg import norm
from sentence_transformers import SentenceTransformer, util

from algorithm.dtree import tree_sim
from algorithm.idf import IDF
from algorithm.utils import line_process, read_stopwords, visualize

app = Flask(__name__)
basepath = os.path.dirname(__file__)
static = 'static'
image_dir = os.path.join(basepath, static)


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


@app.route('/get_text', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':

        s_t = request.form.get('source_text')
        t_t = request.form.get('target_text')

        corpus = []
        cor = line_process(s_t, stopwords)
        corpus.append(cor)
        cor = line_process(t_t, stopwords)
        corpus.append(cor)
        try:
            vectors = idf(corpus)
            tfidf_sim = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
        except:
            tfidf_sim = 0

        query_embedding = sBert.encode(s_t)
        passage_embedding = sBert.encode(t_t)
        nn_cosine_sim = util.pytorch_cos_sim(query_embedding, passage_embedding).numpy()[0][0]
        nn_dot_sim = util.dot_score(query_embedding, passage_embedding).numpy()[0][0]

        doc = HanLP([s_t, t_t])
        t_sim = tree_sim(doc['dep'])
        results = ddp.parse([s_t, t_t])
        result = results[0]
        img = visualize(result['word'], result['head'], result['deprel'], result['postag'])
        s_t_image = os.path.join(image_dir, 's_t.png')
        t_t_image = os.path.join(image_dir, 't_t.png')
        s_image = os.path.join(static, 's_t.png')
        t_image = os.path.join(static, 't_t.png')
        cv2.imwrite(s_t_image, img)
        result = results[1]
        img = visualize(result['word'], result['head'], result['deprel'], result['postag'])
        cv2.imwrite(t_t_image, img)

        sim = '神经网络余弦相似度:' + str(nn_cosine_sim) + '\n'
        sim += '神经网络点积相似度:' + str(nn_dot_sim) + '\n'
        sim += '依存树相似度:' + str(t_sim) + '\n'
        sim += 'TFIDF相似度:' + str(tfidf_sim)[:9] + '\n'
        res_json = {
            'sim': sim,
            'tree_json': doc,
            's_t_image': s_image,
            't_t_image': t_image,
        }

        return render_template('index.html', res_json=res_json, )


if __name__ == '__main__':
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    sBert = SentenceTransformer('algorithm/checkpoints/distiluse-base-multilingual-cased-v1')
    ddp = DDParser(use_pos=True)
    idf = IDF()
    stopwords = read_stopwords('data/StopWords.txt')
    idf.restore('algorithm/checkpoints')
    app.run(host='127.0.0.1', debug=True, port=5020, use_reloader=True)
