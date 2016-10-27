# -*- coding: utf-8 -*-
import re
from zhconvert import ZHConvert
from gliacloud_api_client import Word2Vec

try:
    zh = ZHConvert('http://localhost:9998/pos?wsdl', 'http://localhost:9999/seg?wsdl')
    zh.tw_postag(u'今天天氣真好')
except:
    zh = ZHConvert()
w2v = Word2Vec('zh')


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
        replace('\t', u'。').replace(u'　', '')
    # remove private use area
    string = ''.join([ch for ch in string if ord(ch) < 0xE000 or ord(ch) > 0xF8FF])
    string = re.sub(' +', ' ', string)
    string = re.sub('\(.+?\)', '', string)
    string = re.sub(u'（.+?）', '', string)
    string = re.sub(u'--.+?--', '', string)
    return string


def sent_tokenize(raw_text):
    return re.findall(u'[^。？！；\?\!\;]+', raw_text)


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
