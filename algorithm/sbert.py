# -*- coding: utf-8 -*-
# !@time: 2021/5/13 上午2:57
# !@author: superMC @email: 18758266469@163.com
# !@fileName: sbert.py

from sentence_transformers import SentenceTransformer, util

sBert = SentenceTransformer('algorithm/checkpoints/distiluse-base-multilingual-cased-v1')

if __name__ == '__main__':
    query_embedding = sBert.encode('你好可爱.')
    passage_embedding = sBert.encode('You are so cute.')

    print("余弦相似度:", util.pytorch_cos_sim(query_embedding, passage_embedding))
    print('点积相似度:', util.dot_score(query_embedding, passage_embedding))
