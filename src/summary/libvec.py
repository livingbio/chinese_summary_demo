# -*- coding: utf-8 -*-
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine, cdist
from gensim.models.word2vec import Word2Vec
import numpy as np
import re
from zhconvert import ZHConvert

zh = ZHConvert()

language_code_to_w2v_file = {
    'zh': '/media/wordvec/enwiki.vec.gensim',
    'zh': '/media/wordvec/twwiki.vec.gensim',
    'zh-tw': '/media/wordvec/twwiki.vec.gensim',
    'zh-cn': '/media/wordvec/zhwiki_stanford',
    'ja': '/media/wordvec/jawiki.vec.gensim',
}


class WordVec(object):
    cached = {}

    def __getitem__(self, code):
        if code not in language_code_to_w2v_file:
            raise ValueError('Invalid language code for wordvec: {}'.format(code))
        if code in self.cached:
            return self.cached[code]
        self.cached[code] = Word2Vec.load(language_code_to_w2v_file[code])
        return self.cached[code]

w2v = WordVec()

stop_words = {
    u'也', u'的', u'之', u'「', u'」', u'是', u'讓', u'在', u'下', u'上',
    u'並', u'以', u'和', u'及', u'起', u'將', u'會', u'就', u'因', u'著',
    u'已', u'由', u'有', u'了', u'要', u'尤其是', u'顯得', u'大多', u'這',
    u'各', u'等', u'昨天', u'今天', u'明天', u'今年', u'代表', u'此',
    u'應', u'表示', u'周末', u'以前', u'被', u'另', u'只', u'能', u'昨日',
    u'曾', u'成', u'象徵', u'相當', u'，', u'。', u'不只', u'、', u'；',
    u'又', u'其中', u'恐怕',
}


def to_half_word(text):
    '''Transfer double-width character to single-width character.'''
    return ''.join([chr(ord(ch) - 0xff00 + 0x20)
                    if ord(ch) >= 0xff01 and ord(ch) <= 0xff5e else ch
                    for ch in text])


def tidify(string):
    string = to_half_word(unicode(string)) + ' '
    string = string.replace(u'\u2010', '-').replace(u'\u2012', '-'). \
        replace(u'\u2013', '-').replace(u'\u2014', '-').replace(u'\u2015', '-'). \
        replace(u'\u2018', '\'').replace(u'\u2019', '\'').replace(u'\u201a', '\''). \
        replace(u'\u201b', '\'').replace(u'\u201c', '"').replace(u'\u201d', '"'). \
        replace(u'\u201e', '"').replace(u'\u201f', '"').replace(u'\u2024', '.').strip()
    string = string.replace('\r', '').replace(u'台灣', u'臺灣'). \
        replace(u'台北', u'臺北').replace(u'台中', u'臺中'). \
        replace(u'台南', u'臺南').replace(u'台東', u'臺東'). \
        replace(u'。」', u'。').replace(u'「', u' ')
    string = re.sub('（.+?）', '', re.sub(' +', ' ', re.sub('\(.+?\)', '', string)))
    return string


def similarity(v1, v2):
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        return 1.0 - cosine(v1, v2)
    if len(v1.shape) == 1:
        vv1 = np.array([v1])
    else:
        vv1 = v1
    if len(v2.shape) == 1:
        vv2 = np.array([v2])
    else:
        vv2 = v2
    return (1 - cdist(vv1, vv2, 'cosine')).flatten()


def find_oov(words):
    oov = set()
    for w in words:
        if w not in w2v['zh']:
            oov.add(w)
    return oov


def is_oov(words):
    oov = np.zeros(len(words), dtype=np.bool)
    for i, w in enumerate(words):
        if w not in w2v['zh']:
            oov[i] = True
    return oov


def terms_to_vec(terms, oov_set):
    '''Convert a list of terms to a vector. The vector is the average of each
        term vector.
    Args:
        model (:obj:`Word2Vec`): a word-vector model. It can be pretrained from
            different algorithms, like word2vec, GloVe, FastText, etc.
        terms (:obj:`list` of :obj:`str`): in English, terms are usually equal
            to words. A word is a term. If Chinese, a term usually is composed
            of several words. Terms are used to query a word-vector model. If
            a term is out-of-vocabulary, it is converted to a zero vector.
    Returns:
        np.ndarray: a 300-d vector as the average of all word vectors.
    '''
    v = np.zeros(w2v['zh'].layer1_size, dtype=np.float32)
    for word in terms:
        if word not in w2v['zh']:
            oov_set.add(word)
            # print 'OOV:', word
            continue
        v += w2v['zh'][word]
    return v / len(terms)


def sentence_vector(sents, ret_oov=False):
    '''Convert a list of sentences to a matrix.
    Args:
        model (:obj:`Word2Vec`): a word-vector model. It can be pretrained from
            different algorithms, like word2vec, GloVe, FastText, etc.
        segment (function): segmentation function.
        sents (:obj:`list` of :obj:`str`): a list of sentences.
    Returns:
        np.ndarray: a Nx300 matrix contains N vectors for each sentence.
    '''
    oov_set = set()
    vec = np.empty((len(sents), w2v['zh'].layer1_size))
    for i, s in enumerate(sents):
        words = [w for w in zh.tw_segment(tidify(s).lower())]
        vec[i, :] = terms_to_vec(words, oov_set)
    if ret_oov:
        return normalize(vec), oov_set
    return normalize(vec)
