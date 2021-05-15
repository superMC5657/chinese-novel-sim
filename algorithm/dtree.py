# -*- coding: utf-8 -*-
# !@time: 2021/5/13 下午9:21
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dtree.py


import json

import cv2
import datasketch
import hanlp
import treesimi as ts

from algorithm.utils import visualize


def dlp(ddp, text, image_path):
    results = ddp.parse(text)
    result = results[0]
    data = visualize(result['word'], result['head'], result['deprel'], result['postag'])
    # or data = result['visual']
    cv2.imwrite(image_path, data)


def tree_sim(deprels):
    cfg = {
        'use_trunc_leaves': True,
        'use_drop_nodes': False,
        'use_replace_attr': False
    }
    mhash = []
    for deprel in deprels:
        adjac = [(index + 1, head, dep) for index, (head, dep) in enumerate(deprel)]
        nested = ts.adjac_to_nested_with_attr(adjac)
        nested = ts.remove_node_ids(nested)
        shingled = ts.shingleset(nested, **cfg)
        stringified = [json.dumps(tree).encode('utf-8') for tree in shingled]
        m = datasketch.MinHash(num_perm=256)
        for s in stringified:
            m.update(s)
        mhash.append(m)
    return mhash[0].jaccard(mhash[1])


# def tree_sim_template():
#     dat = conllu.parse(open("data/de_hdt-ud-dev.conllu").read())
#     cfg = {
#         'use_trunc_leaves': True,
#         'use_drop_nodes': False,
#         'use_replace_attr': False
#     }
#
#     mhash = []
#     for i in (54, 51, 56, 57, 58):
#         adjac = [(t['id'], t['head'], t['deprel']) for t in dat[i]]
#         nested = ts.adjac_to_nested_with_attr(adjac)
#         nested = ts.remove_node_ids(nested)
#         shingled = ts.shingleset(nested, **cfg)
#         stringified = [json.dumps(tree).encode('utf-8') for tree in shingled]
#         m = datasketch.MinHash(num_perm=256)
#         for s in stringified:
#             m.update(s)
#         mhash.append(m)
#     for i in range(len(mhash)):
#         print(mhash[0].jaccard(mhash[i]))


if __name__ == '__main__':
    s1 = '莲步微移，名为萧薰儿的少女行到魔石碑之前，小手伸出，镶着黑金丝的紫袖滑落而下，露出一截雪白娇嫩的皓腕，然后轻触着石碑'
    s2 = '这名紫裙少女，论起美貌与气质来，比先前的萧媚，无疑还要更胜上几分，也难怪在场的众人都是这般动作。'

    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)  # 世界最大中文语料库
    doc = HanLP([s1])
    doc.pretty_print()
    # results = HanLP([s1, s2])
    # tree_sim(results['dep'])
    # tree_sim_template()

    #
    # ddp = DDParser(use_pos=True)
    # results = ddp.parse([s1, s2])
    # result = results[0]
    # img = visualize(result['word'], result['head'], result['deprel'], result['postag'])
    # cv2.imwrite('test.jpg', img)
