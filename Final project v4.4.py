# Author: Group 9 untitled
# Date: 2020/7/12 - 2020/7/15

import requests
import bs4
import xlsxwriter
from urllib import request
import json
from pandas.core.frame import DataFrame
from snownlp import SnowNLP
import re
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import jieba
import operator
from gensim import corpora, models
import math

SPIDER_PATH = 'Result.csv'
COMPRESSED_PATH = 'Compressedresult.csv'
HEADER = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                            '65.0.3325.1''81 Safari/537.36'}
URL = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={productId}" \
        "&score={score}&sortType=5&page={page}&pageSize=10&isShadowSku=0&rid=0&fold=1"
DECODE = "GBK"


def fetch_json(productId, page, score) -> dict:
    url = URL.format(
        productId=str(productId),
        page=str(page),
        score=str(score),
    )
    req = request.Request(url, headers=HEADER)
    rep = request.urlopen(req).read().decode(DECODE, errors="ignore")
    dict_ = re.findall(r"\{.*\}", rep)[0]
    return json.loads(dict_)


def get_data(productId, sheetName, fileName):
    all_comment = 0
    p = 1
    sum = 1
    while sum > 0:
        sum = 0
        comment_dict = fetch_json(productId, page=str(p), score=0)
        p += 1
        for i in comment_dict["comments"]:
            sum += 1
            write_col = [i['content']]
            df = pd.DataFrame(columns=(write_col))
            df.to_csv(fileName, line_terminator="\n", index=False, mode='a', encoding='gb18030')
            all_comment += 1


# 机械压缩部分函数
def judge_repeat(L1, L2):
    if len(L1) != len(L2):
        return False
    else:
        return operator.eq(L1, L2)


def machine_compressed(commentList):
    L1 = []
    L2 = []
    compressList = []
    for letter in commentList:
        if len(L1) == 0:
            L1.append(letter)
        else:
            if L1[0] == letter:
                if len(L2) == 0:
                    L2.append(letter)
                else:
                    if judge_repeat(L1, L2):
                        L2.clear()
                        L2.append(letter)
                    else:
                        compressList.extend(L1)
                        compressList.extend(L2)
                        L1.clear()
                        L2.clear()
                        L1.append(letter)

            else:
                if judge_repeat(L1, L2) and len(L2) >= 2:
                    compressList.extend(L1)
                    L1.clear()
                    L2.clear()
                    L1.append(letter)
                else:
                    if len(L2) == 0:
                        L1.append(letter)
                    else:
                        L2.append(letter)
    else:
        if judge_repeat(L1, L2):
            compressList.extend(L1)
        else:
            compressList.extend(L1)
            compressList.extend(L2)
    L1.clear()
    L2.clear()
    return compressList


def sentiment_analysis(datalist):
    sentiment_dic = {}
    for text in datalist:
        s = SnowNLP(text)
        sentiment_dic[text] = s.sentiments
        # print(text[:10] + " {}".format(s.sentiments))
    return sentiment_dic


def network():
    s = pd.read_csv('Result.csv', encoding=DECODE, header=None)
    data_processed = pd.DataFrame(s[0].unique())
    string_data = ''.join(list(data_processed[0]))
    num = 40
    G = nx.Graph()
    plt.figure(figsize=(20, 14))
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    pattern = re.compile(u'\t|\。|，|：|；|！|\）|\（|\?|"')
    string_data = re.sub(pattern, '', string_data)
    seg_list_exact = jieba.cut(string_data, cut_all=False)
    object_list = []
    stop_words = list(open('stopwords.txt', 'r', encoding='utf-8').read())
    stop_words.append("\n")
    for word in seg_list_exact:
        if word not in stop_words:
            object_list.append(word)
    word_counts = collections.Counter(object_list)
    word_counts_top = word_counts.most_common(num)
    word = pd.DataFrame(word_counts_top, columns=['关键词', '次数'])
    word_T = pd.DataFrame(word.values.T, columns=word.iloc[:, 0])
    net = pd.DataFrame(np.mat(np.zeros((num, num))), columns=word.iloc[:, 0])
    k = 0
    object_list2 = []
    for i in range(len(string_data)):
        if string_data[i] == '\n':
            seg_list_exact = jieba.cut(string_data[k:i], cut_all=False)
            for words in seg_list_exact:
                if words not in stop_words:
                    object_list2.append(words)
            k = i + 1
    word_counts2 = collections.Counter(object_list2)
    word_counts_top2 = word_counts2.most_common(num)
    word2 = pd.DataFrame(word_counts_top2)
    word2_T = pd.DataFrame(word2.values.T, columns=word2.iloc[:, 0])
    relation = list(0 for x in range(num))
    for j in range(num):
        for p in range(len(word2)):
            if word.iloc[j, 0] == word2.iloc[p, 0]:
                relation[j] = 1
                break
    for j in range(num):
        if relation[j] == 1:
            for q in range(num):
                if relation[q] == 1:
                    net.iloc[j, q] = net.iloc[j, q] + word2_T.loc[1, word_T.iloc[0, q]]
    n = len(word)
    for i in range(n):
        for j in range(i, n):
            G.add_weighted_edges_from([(word.iloc[i, 0], word.iloc[j, 0], net.iloc[i, j])])
    try:
        nx.draw_networkx(G, pos=nx.shell_layout(G),
                     width=[float((v['weight'] - 53) / 100) for (r, c, v) in G.edges(data=True)],
                     edge_color=np.arange(len(G.edges)),
                     node_size=[float((net.iloc[i, i] - 85) * 10) for i in np.arange(20)],
                     node_color=np.arange(40))
    except ValueError:
        nx.draw_networkx(G, pos=nx.shell_layout(G),
                         width=[float((v['weight'] - 53) / 300) for (r, c, v) in G.edges(data=True)],
                         edge_color=np.arange(len(G.edges)),
                         node_size=[float((net.iloc[i, i] - 85) * 10) for i in np.arange(40)],
                         node_color=np.arange(40))
    plt.axis('off')
    plt.savefig("NetWork.png")
    # plt.show()


def LDA(data_final_list):
    data_final = []
    for data in data_final_list:
        cut = jieba.cut(data, cut_all=False, HMM=False)
        data_final.append(' '.join(cut))
    pddata = pd.DataFrame(data_final)
    # print(pddata[0])
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = list(f.readlines())
        # print(stopwords)
    stopword = []
    for i in stopwords:
        stopword.append(i[:-1])
    # print(stopword)
    stopword = [' ', '', '　'] + list(stopword)
    pddata[1] = pddata[0].apply(lambda s: s.split(' '))
    pddata[2] = pddata[1].apply(lambda x: [i for i in x if i not in stopword])
    # print(pddata[2])

    dictionary = corpora.Dictionary(pddata[2])
    corpus = [dictionary.doc2bow(data) for data in pddata[2]]
    model = models.LdaModel(corpus, id2word=dictionary, iterations=500, num_topics=3, alpha='auto')
    for i in range(3):
        print(model.print_topic(i))


def transform(filename):
    s = pd.read_csv(filename, encoding='gb18030', header=None)
    data_processed = pd.DataFrame(s[0].unique())
    string_data = ''.join(list(data_processed[0]))
    file = open("rost.txt", 'a', encoding='ansi')
    for text in string_data:
        file.write(text)
    file.close()


if __name__ == '__main__':
    # 数据爬取
    # maxpage = eval(input('需要爬取的商品个数'))
    # item_name = input('需要爬取的商品名称')
    # url1 = 'https://search.jd.com/Search?keyword=' + item_name + '&enc=utf-8&wq=' + item_name
    # r = requests.get(url=url1, headers=HEADER)
    # soup = bs4.BeautifulSoup(r.text, "html5lib")
    # list1 = soup.find_all('i', {'class': "promo-words"})
    # for i in range(maxpage):
    #     try:
    #         print("正在爬取第" + str(i + 1) + "件商品评论")
    #         tag = list1[i]
    #         id = tag['id'][5:]
    #         get_data(id, item_name, 'Result.csv')
    #     except:
    #         print("爬取失败")
    # print("爬取完成")

    # 网路语义分析
    network()
    print("网路语义分析完成")

    # 数据清洗部分
    # 朴素去重
    s = pd.read_csv(SPIDER_PATH, encoding=DECODE, header=None)
    len1 = len(s)
    data_processed = pd.DataFrame(s[0].unique())
    len2 = len(data_processed)
    print("原有%d条评论" % len1)
    print("现有%d条评论" % len2)
    print("删除了%d条重复评论" % (len1 - len2))
    data_processed_list = data_processed[0].values.tolist()
    print("去重完成")

    # 调用机械压缩
    data_compressed_list = machine_compressed(data_processed_list)
    # print(data_compressed_list)
    print("机械压缩完成")

    # 短词删除
    data_final_list = []
    for data in data_compressed_list:
        strdata = str(data)
        strdata = re.sub("[\s+\.\!V,$%^*(+\"\"]+|[+!,.?、~@#$%......&*();`:]+", "", strdata)
        if len(strdata) <= 4:
            pass
        else:
            data_final_list.append(strdata)

    # 文本评论分词
    print("====正在分词====")
    data_final = []
    for data in data_final_list:
        cut = jieba.cut(data, cut_all=False, HMM=False)
        data_final.append(' '.join(cut))

        '''另外两种分词模式，可以放在模型比较中来比较效果
        cut = jieba.cut(s, cut_all=True)
        print(','.join(cut))

        cut = jieba.cut(s, cut_all=True, HMM=True)
        print(','.join(cut))'''
    # print(data_final_list)

    # 分词
    final_df = DataFrame(data_final_list)
    final_df.to_csv(COMPRESSED_PATH, encoding=DECODE, header=None)

    # 情感分析模型--SnowNlp
    print("====SnowNlp情感分析====")
    dic = sentiment_analysis(data_final_list)
    good = []
    bad = []
    k = list(dic.keys())
    for sentence in k:
        if dic[sentence] > 0.90:
            good.append(sentence)
        else:
            bad.append(sentence)
    print(good)
    print(bad)

    # LDA主题分析
    print("====LDA====")
    print("正面主题分析")
    LDA(good)
    print("============")
    print("负面主题分析")
    LDA(bad)

    # 利用ROST前的数据变换，将csv转换成txt
    print("====正在转换为txt====")
    transform('Result.csv')
