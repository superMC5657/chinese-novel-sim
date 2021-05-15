# -*- coding: utf-8 -*-
# !@time: 2021/5/14 上午12:11
# !@author: superMC @email: 18758266469@163.com
# !@fileName: utils.py

import os
import re
import jieba.analyse
import jieba
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

font = font_manager.FontProperties(
    fname='data/SourceHanSansCN-Regular.otf')

# 可视化依存树
def visualize(word, head, deprel, postag):
    """
    Visualize the dependency.

    Args:
        word: list[str], the tokenized words.
        head: list[int], the head ids.
        deprel: list[str], the dependency relation.
        postag: list[str], Part-of-speech tag

    Returns:
        data: a numpy array, use cv2.imshow to show it or cv2.imwrite to save it.
    """
    nodes = ['ROOT'] + word
    x = list(range(len(nodes)))
    y = [0] * (len(nodes))
    fig, ax = plt.subplots()
    # control the picture size
    max_span = max([abs(i + 1 - j) for i, j in enumerate(head)])
    fig.set_size_inches((len(nodes), max_span / 2))
    # set the points
    plt.scatter(x, y, c='w')

    for i in range(len(nodes)):
        txt = nodes[i]
        xytext = (i, 0)
        if i == 0:
            # set 'ROOT'
            ax.annotate(
                txt,
                xy=xytext,
                xycoords='data',
                xytext=xytext,
                textcoords='data',
            )
        else:
            tag = postag[i - 1]
            xy = (head[i - 1], 0)
            rad = 0.5 if head[i - 1] < i else -0.5
            # set the word
            ax.annotate(
                txt,
                xy=xy,
                xycoords='data',
                xytext=(xytext[0] - 0.1, xytext[1]),
                textcoords='data',
                fontproperties=font,
            )
            # set the tag
            ax.annotate(
                tag,
                xy=xy,
                xycoords='data',
                xytext=(xytext[0] - 0.1, xytext[1] - 0.2),
                textcoords='data',
                fontproperties=font
            )

            # draw the curve
            ax.annotate(
                "",
                xy=xy,
                xycoords='data',
                xytext=xytext,
                textcoords='data',
                arrowprops=dict(
                    arrowstyle="<-",
                    shrinkA=12,
                    shrinkB=12,
                    color='blue',
                    connectionstyle="arc3,rad=%s" % rad,
                ),
            )
            # set the deprel label. Calculate its position by the radius
            text_x = min(i, head[i - 1]) + abs((i - head[i - 1])) / 2 - 0.2
            text_y = abs((i - head[i - 1])) / 4
            ax.annotate(
                deprel[i - 1],
                xy=xy,
                xycoords='data',
                xytext=[text_x, text_y],
                textcoords='data')

    # control the axis
    plt.axis('equal')
    plt.axis('off')

    # save to numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] +
                        (3,))[:, :, ::-1]
    return data

# 读取小说
def read_novel(txt_dir_path, stopwords):
    corpus = []
    txt_paths = [os.path.join(txt_dir_path, _) for _ in os.listdir(txt_dir_path)]
    for txt_path in txt_paths:
        with open(txt_path, 'r+', encoding='utf-8') as f:
            for line in f.readlines():
                cor = line_process(line, stopwords)
                if cor:
                    corpus.append(cor)
    return corpus

# 文本行处理
def line_process(line, stopwords):
    line = line.strip()
    line = re.sub(r'%[0-9a-zA-Z][0-9a-zA-Z]', '', line)
    words = jieba.analyse.textrank(line, allowPOS=('n', 'nz', 'a', 'v', 'vn'))
    if not words:
        return False
    cor = []
    for word in words:
        if word not in stopwords:
            cor.append(word)
    cor = " ".join(cor)
    return cor

# 读取停止词
def read_stopwords(txt_path):
    stopwords = [line.strip() for line in open(txt_path, encoding='UTF-8').readlines()]
    return stopwords

# 创建文件夹
def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
