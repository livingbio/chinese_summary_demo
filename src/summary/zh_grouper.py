# -*- coding: utf-8 -*-
import numpy as np
import re
from os import path
from zh_lib import tidify, sent_tokenize, w2v
from summarizer import similarity
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.externals import joblib

cls_fn = path.join(path.dirname(__file__), 'zh_group_classifier.joblib')
reg_fn = path.join(path.dirname(__file__), 'zh_group_regressor.joblib')
rf_classifier, rf_regressor = joblib.load(cls_fn), joblib.load(cls_fn)


def train_classifier(txt):
    X = np.zeros((len(txt), len(txt[0].feat)))
    for i, t in enumerate(txt):
        X[i, :] = t.feat
    y = np.array([t.n > 1 for t in txt], dtype=bool)
    rf = RandomForestClassifier().fit(X, y)
    joblib.dump(rf, 'rf_classifier.joblib')
    return rf.score(X, y)


def train_regressor(txt):
    X = np.zeros((len(txt), len(txt[0].feat)))
    for i, t in enumerate(txt):
        X[i, :] = t.feat
    y = np.array([t.n for t in txt], dtype=float)
    rf = RandomForestRegressor().fit(X, y)
    joblib.dump(rf, 'rf_regressor.joblib')
    return rf.score(X, y)


zhnum = {
    u'一': 1,
    u'二': 2,
    u'三': 3,
    u'四': 4,
    u'五': 5,
    u'六': 6,
    u'七': 7,
    u'八': 8,
    u'九': 9,
}


def find_continue_seq(numbers):
    if not numbers:
        return 1, 0, 0
    group = []
    start, end = 0, 0
    for i in range(1, len(numbers)):
        if numbers[i] <= numbers[i - 1] + 1:
            end = i
        else:
            group.append((numbers[start], numbers[end], numbers[end] - numbers[start] + 1))
            start, end = i, i
    group.append((numbers[start], numbers[end], numbers[end] - numbers[start] + 1))
    start, end, length = group[np.argmax([g[2] for g in group])]
    if length < 3 or length > 20 or end > 20 or start > 5:
        return 1, 0, 0
    if start == 0:
        length, start = length - 1, 1
    return length, start, end


def zh_number_seq(text):
    text = '\n'.join([s[:10] for s in text.split('\n')])
    text = translate_zh_number(re.sub('[0-9]+', '', text))
    numbers = list(set(map(int, re.findall('[0-9]+', text))))
    return find_continue_seq(sorted(numbers))


def en_number_seq(text):
    text = '\n'.join([s[:10] for s in text.split('\n')])
    numbers = list(set(map(int, re.findall('[0-9]+', text))))
    return find_continue_seq(sorted(numbers))


def translate_zh_number(string):
    for k in zhnum:
        string = string.replace(k, str(zhnum[k]))   # 十三 -> 十3, 二十一 -> 2十1
    string = re.sub(u'([1-9])十([1-9])', '\\1\\2', string)  # 2十1 -> 21
    string = re.sub(u'十([1-9])', '1\\1', string)   # 十3 -> 13
    return string.replace(u'十', '10')


class TextGrouper(object):
    def __init__(self, raw_text):
        self.raw_text = re.sub('\n\n+', '\n@@@\n', raw_text.replace('\r', ''))
        tidy_text = tidify(self.raw_text.replace(u'\n', u'。\n'))
        self.tidy_text = re.sub(u'^。+', '', re.sub(u'。。+', u'。', tidy_text))
        paragraph = [p for p in tidy_text.split(u'@@@。') if len(p) > 4]
        self.paragraph = []
        sents = []
        for p in paragraph:
            para = [s for s in sent_tokenize(p) if len(s) > 4]
            i = len(sents)
            sents.extend(para)
            j = len(sents)
            self.paragraph.append(np.arange(i, j))
        self.sents = np.array(sents)
        self.sent_numbers = None
        self.vector = w2v.sentvec(self.sents)
        self.gen_feature()
        predc = rf_classifier.predict(np.expand_dims(self.feat, 0))[0]
        self.predn = max(int(rf_regressor.predict(np.expand_dims(self.feat, 0))[0]), 1)
        if not predc or self.predn <= 2:
            self.predn = 1
        elif self.predn < 5:
            self.predn += 1

    def get_number_seq(self):
        seq1 = zh_number_seq(self.raw_text)
        seq2 = en_number_seq(self.raw_text)
        if not hasattr(self, 'predn'):
            if seq1[0] > seq2[0]:
                return seq1, 'zh'
            else:
                return seq2, 'en'
        if self.predn == 1:
            return (1, 0, 0), None
        if abs(seq1[0] - self.predn) < abs(seq2[0] - self.predn):
            return seq1, 'zh'
        else:
            return seq2, 'en'

    def get_gap_threshold(self, target_n=None):
        if len(self.vector) == 1:
            return 1.0
        if target_n is None:
            target_n = self.predn
        dist = []
        for v1, v2 in zip(self.vector[:-1], self.vector[1:]):
            dist.append(1 - similarity(v1, v2) ** 15)
        dist = np.sort(np.array(dist))
        return (dist[-target_n - 1] + dist[-target_n]) / 2

    def group_by_paragraph(self):
        return self.paragraph

    def group_by_gap(self, threshold=None):
        if threshold is None:
            threshold = self.get_gap_threshold()
        groups = []
        cluster = [0]
        i = 1
        for v1, v2 in zip(self.vector[:-1], self.vector[1:]):
            curr_dist = 1 - similarity(v1, v2) ** 15
            if curr_dist > threshold:
                groups.append(np.array(cluster))
                cluster = [i]
            else:
                cluster.append(i)
            i += 1
        groups.append(np.array(cluster))
        return groups

    def group_by_number(self):
        (length, start, end), lang = self.get_number_seq()
        if lang is None or length <= 2:
            return None
        elif lang == 'en':
            senthead = [s[:10] for s in self.sents]
        else:
            senthead = [translate_zh_number(re.sub('[0-9]+', '', s[:10]))
                        for s in self.sents]
        numbers = [-1] * len(senthead)
        last = None
        for i, s in enumerate(senthead):
            x = re.findall('[0-9]+', s)
            if not x:
                continue
            num = int(x[0])
            if last is None:
                if num == start or num == end:
                    numbers[i], last = num, num
            else:
                if abs(num - last) == 1:
                    numbers[i], last = num, num
        self.sent_numbers = np.where(np.array(numbers) > 0)[0]
        cluster = []
        groups = []
        for num, i in zip(numbers, range(len(self.sents))):
            if num > 0:
                groups.append(np.array(cluster))
                cluster = []
            else:
                cluster.append(i)
        groups.append(np.array(cluster))
        if len(groups) <= 2:
            return None
        return groups

    def group(self):
        groups = self.group_by_number()
        if groups is not None:
            return groups
        groups = self.group_by_paragraph()
        if abs(len(groups) - self.predn) <= 1:
            return groups
        return self.group_by_gap()

    def get_n_sent_gap(self, threshold=0.6):
        n_cluster = 1
        for v1, v2 in zip(self.vector[:-1], self.vector[1:]):
            curr_dist = 1 - similarity(v1, v2) ** 15
            if curr_dist > threshold:
                n_cluster += 1
        return n_cluster

    def gen_feature(self):
        feat = []
        feat.append(len(self.sents))
        feat.append(len(self.raw_text))
        feat.append(self.get_n_sent_gap())
        feat.append(self.get_number_seq()[0][0])
        self.feat = np.array(feat)
