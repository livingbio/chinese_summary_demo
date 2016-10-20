# -*- coding: utf-8 -*-
import numpy as np
import re
from zh_tree import ChineseTree
from gliacloud_api_client import Word2Vec
from summarizer import similarity, dynaprog_summarizer, cluster_summarizer, maximal_summarizer
from zhconvert import ZHConvert
from datetime import datetime as dt

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
        replace(u'【', '').replace(u'】', '').replace(u'「', '').replace(u'」', ''). \
        replace('\t', u'。')
    # remove private use area
    string = ''.join([ch for ch in string if ord(ch) < 0xE000 or ord(ch) > 0xF8FF])
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


def adjust_by_nouns(score, sentences, proper_nouns, growth=1.05):
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


dont_split_word = {
    u'也', u'但', u'仍', u'較', u'再', u'為',
}


def split_sentence(sentence):
    tagtext = zh.tw_postag(sentence)
    split_point = []
    for i in range(2, len(tagtext)):
        t1 = universal_tagset[tagtext[i - 2][1]]
        t2 = universal_tagset[tagtext[i - 1][1]]
        t3 = universal_tagset[tagtext[i][1]]
        w1, w2, w3 = tagtext[i - 2][0], tagtext[i - 1][0], tagtext[i][0]
        if (t1 in ('VERB', 'NOUN') and t2 == '.' and t3 in ('ADV', 'ADP')) or \
            (t1 == 'PRT' and t2 == '.' and t3 == 'NOUN') and w3 not in dont_split_word:
            split_point.append((sentence.find(w1 + w2 + w3) + len(w1), t1))
    if not split_point:
        return [sentence]  # no way to split the sentence

    splitted = []
    for i, (cut, cut_pos) in enumerate(split_point[::-1]):
        if cut < 20 or len(sentence) - cut < 20:
            continue
        if cut_pos == 'PRT' and i < len(split_point) - 1:
            cut2 = next(p for p, _ in split_point[::-1] if p < cut)
        else:
            cut2 = cut
        splitted.append(sentence[(cut + 1):])
        sentence = sentence[:cut2]
    splitted.append(sentence)
    return splitted[::-1]


def newline_hint(string):
    string = re.sub(u'([A-Za-z\.]) ([A-Za-z\.])', u'\\1_\\2', string.replace('_', ''))
    string = re.sub(u'([A-Za-z\.]) ([A-Za-z\.])', u'\\1_\\2', string)
    seg = [w.replace('_', ' ') for w in zh.tw_segment(string)]
    return '_'.join(seg)


def chunking_sent(sentence, forceFirstSubSent=False):
    start = dt.now()
    length = zhlen(sentence)
    if length < 30:
        return [newline_hint(sentence)]
    if length < 70:
        return ChineseTree(sentence).chunking()
    chunks = []
    split_sents = split_sentence(sentence)
    if forceFirstSubSent:
        return ChineseTree(split_sents[0]).chunking()
    for subsent in split_sents:
        if zhlen(subsent) < 30:
            chunks.append(newline_hint(subsent))
        else:
            chunks += ChineseTree(subsent).chunking()
    return chunks


def shorten_sents(summary, ref_vec=None, special_nouns=None):
    shorten = []
    first_sent = True
    for s in summary:
        if ref_vec is None:
            ref_vec = w2v.sentvec(s)[0]
        chunks = chunking_sent(s, first_sent)
        first_sent = False
        if chunks:
            score = similarity(w2v.sentvec([ch.replace('_', '') for ch in chunks]), ref_vec)
            if special_nouns:
                score = adjust_by_nouns(score, chunks, special_nouns)
            score = adjust_by_len(score, chunks)
            shorten.append([ch for ch in zip(score, chunks) if not np.isnan(ch[0])])
        else:
            shorten.append([(1.0, newline_hint(s))])
    return np.array(shorten)


def summary_text(raw_text, n_summary=5, algorithm=2, shorten=True):
    '''Given a text string, split it into sentences.
    Find `n_summary` sentences to summarize the text.
    Args:
        raw_text (str): a string consisted of several sentences.
        n_summary (str): the number of sentences to be put into summary.
        algorithm (int): choose an algorithm to summarize.
        shorten (bool): return shorten summaries if shorten=True
    Returns:
        :obj:`np.array` of :obj:`str`: a list of summary sentences.
    '''
    raw_text = tidify(raw_text.replace(u'\n\n', u'。\n\n'))
    special_nouns = find_special_nouns(raw_text)
    # print '>>> Proper Nouns', '/'.join(special_nouns)

    sents = np.array([s for s in sent_tokenize(raw_text) if zhlen(s) > 10])
    score_reward = adjust_by_nouns([1.0] * len(sents), sents, special_nouns, growth=1.01)
    # score_reward = np.ones(len(sents))
    sents_vector = w2v.sentvec(sents)
    article_vector = np.sum(sents_vector, axis=0)

    if algorithm == 0:
        index = dynaprog_summarizer(sents_vector, article_vector)
    elif algorithm == 1:
        if len(sents) <= n_summary:
            index = np.arange(len(sents))
        else:
            index = cluster_summarizer(sents_vector, article_vector, n_summary, score_reward)
    elif algorithm == 2:
        index = maximal_summarizer(sents_vector, article_vector, n_summary, score_reward)

    if shorten:
        return shorten_sents(sents[index])
    else:
        summary = []
        for i in index:
            summary.append((similarity(sents_vector[i], article_vector), newline_hint(sents[i])))
        return np.array(summary)
