# -*- coding: utf-8 -*-
import numpy as np
import re
from zh_tree import ChineseTree
from word2vec_client import Word2Vec
from summarizer import similarity, dynaprog_summarizer, cluster_summarizer, maximal_summarizer
from zhconvert import ZHConvert

try:
    zh = ZHConvert('http://localhost:9998/pos?wsdl', 'http://localhost:9999/seg?wsdl')
    zh.tw_postag(u'今天天氣真好')
except:
    zh = ZHConvert()
w2v = Word2Vec('zh')
summary_data = {}
universal_tagset = {
    'AD': 'ADV',
    'AS': 'PRT',
    'BA': 'X',
    'CC': 'CONJ',
    'CD': 'NUM',
    'CS': 'CONJ',
    'DEC': 'PRT',
    'DEG': 'PRT',
    'DER': 'PRT',
    'DEV': 'PRT',
    'DT': 'DET',
    'ETC': 'PRT',
    'FW': 'X',
    'IJ': 'X',
    'JJ': 'ADJ',
    'LB': 'X',
    'LC': 'PRT',
    'M': 'NUM',
    'MSP': 'PRT',
    'NN': 'NOUN',
    'NR': 'NOUN',
    'NT': 'NOUN',
    'OD': 'NUM',
    'ON': 'X',
    'P': 'ADP',
    'PN': 'PRON',
    'PU': '.',
    'SB': 'X',
    'SP': 'PRT',
    'VA': 'VERB',
    'VC': 'VERB',
    'VE': 'VERB',
    'VV': 'VERB',
    'X': 'X',
}


def zhlen(text):
    try:
        return len(text.replace('_', '').encode('big5')) // 2.0
    except:
        return len(text.replace('_', ''))


def to_half_word(text):
    '''Transfer double-width character to single-width character.'''
    return ''.join([chr(ord(ch) - 0xff00 + 0x20)
                    if ord(ch) >= 0xff01 and ord(ch) <= 0xff5e else ch
                    for ch in text])


def tidify(string):
    string = to_half_word(string) + ' '
    string = string.replace(u'\u2010', '-').replace(u'\u2012', '-'). \
        replace(u'\u2013', '-').replace(u'\u2014', '-').replace(u'\u2015', '-'). \
        replace(u'\u2018', '\'').replace(u'\u2019', '\'').replace(u'\u201a', '\''). \
        replace(u'\u201b', '\'').replace(u'\u201c', '"').replace(u'\u201d', '"'). \
        replace(u'\u201e', '"').replace(u'\u201f', '"').replace(u'\u2024', '.'). \
        replace(u'\n', '').replace(u'\r', '').replace(u'《', '').replace(u'》', ''). \
        replace(u'【', '').replace(u'】', '').replace(u'「', '').replace(u'」', '')
    string = re.sub(' +', ' ', string)
    string = re.sub('\(.+?\)', '', string)
    string = re.sub(u'（.+?）', '', string)
    string = re.sub(u'--.+?--', '', string)
    return string


def sent_tokenize(raw_text):
    return re.findall(u'[^。？！；\?\!\;]+', raw_text)


def find_special_nouns(raw_text):
    proper_noun = set()
    for w in w2v.oovword(raw_text):
        cnt = raw_text.count(w)
        cntlower = raw_text.count(w.lower())
        # print 'OOV', w, cnt, cntlower, cnt > 4, (cnt - cntlower) > 2
        if any([c.isupper() for c in w]) and cnt > 4 and (cnt - cntlower) > 2:
            proper_noun.add(w)
        elif len(w) > 1 and cnt > 8 and all([ord(ch) > 256 for ch in w]):
            proper_noun.add(w)
    return proper_noun


def adjust_by_nouns(score, sentences, proper_nouns=None, growth=1.05):
    if not proper_nouns:
        proper_nouns = summary_data['special_nouns']
    for i, s in enumerate(sentences):
        words = s.split()
        n_proper_nouns = sum([w in proper_nouns for w in words])
        reward = growth ** n_proper_nouns
        score[i] *= reward
    return score


def adjust_by_len(score, sentences, limit=30, decay=0.998):
    for i, s in enumerate(sentences):
        l = zhlen(s)
        if l > limit + 10:
            panelty = decay ** (l - limit - 10)
        elif l < limit:
            panelty = decay ** (limit - l)
        else:
            panelty = 1.0
        score[i] *= panelty
    return score


def summary_text(raw_text, n_summary=5, algorithm=2):
    '''Given a text string, split it into sentences.
    Find `n_summary` sentences to summarize the text.
    Args:
        raw_text (str): a string consisted of several sentences.
        n_summary (str): the number of sentences to be put into summary.
        algorithm (int): choose an algorithm to summarize.
    Returns:
        :obj:`np.array` of :obj:`str`: a list of summary sentences.
    '''
    raw_text = tidify(raw_text.replace(u'\n\n', u'。\n\n'))
    summary_data['special_nouns'] = find_special_nouns(raw_text)
    # print '>>> Proper Nouns', '/'.join(summary_data['special_nouns'])

    sents = np.array([s for s in sent_tokenize(raw_text) if zhlen(s) > 10])
    score_reward = adjust_by_nouns([1.0] * len(sents), sents, growth=1.01)
    # score_reward = np.ones(len(sents))
    sents_vector = w2v.sentvec(sents)
    article_vector = np.sum(sents_vector, axis=0)
    summary_data['article_vector'] = article_vector

    if algorithm == 0:
        index = dynaprog_summarizer(sents_vector, article_vector)
    elif algorithm == 1:
        if len(sents) <= n_summary:
            index = np.arange(len(sents))
        else:
            index = cluster_summarizer(sents_vector, article_vector, n_summary, score_reward)
    elif algorithm == 2:
        index = maximal_summarizer(sents_vector, article_vector, n_summary, score_reward)
    summary_data['summary_vector'] = sents_vector[index]
    return np.array(sents[index])


def shorten_sents(summary):
    shorten = []
    for vec, s in zip(summary_data['summary_vector'], summary):
        l = zhlen(s)
        if l < 30:
            shorten.append([(1.0, s)])
            continue
        if l > 70:
            tagtext = zh.tw_postag(s)
            split_point = []
            for i in range(2, len(tagtext)):
                t1 = universal_tagset[tagtext[i - 2][1]]
                t2 = universal_tagset[tagtext[i - 1][1]]
                t3 = universal_tagset[tagtext[i][1]]
                if (t1 in ('VERB', 'NOUN') and t2 == '.' and t3 in ('ADV', 'ADP')) or \
                    (t1 == 'PRT' and t2 == '.' and t3 == 'NOUN'):
                    w1, w2, w3 = tagtext[i - 2][0], tagtext[i - 1][0], tagtext[i][0]
                    split_point.append(s.find(w1 + w2 + w3) + len(w1))
            if split_point:
                p = split_point[-1]  # split_point[np.argmin(np.abs(np.array(split_point) - len(s) // 2))]
                if len(split_point) > 1 and (p < 20 or len(s) - p < 20):
                    p = split_point[-2]
                chunks = ChineseTree(s[:p]).chunking() + ChineseTree(s[(p + 1):]).chunking()
            else:
                chunks = ChineseTree(s).chunking()
        else:
            chunks = ChineseTree(s).chunking()
        score = similarity(w2v.sentvec([ch.replace('_', '') for ch in chunks]), summary_data['article_vector'])
        score = adjust_by_nouns(score, chunks)
        score = adjust_by_len(score, chunks)
        shorten.append(zip(score, chunks))
    return np.array(shorten)
