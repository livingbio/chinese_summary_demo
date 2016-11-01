# -*- coding: utf-8 -*-
from nltk.chunk.regexp import ChunkRule, ChinkRule
from nltk.chunk import tagstr2tree
from nltk import RegexpChunkParser
from zh_lib import w2v, zh, zhlen
from summarizer import similarity
import re
import numpy as np

freq_words = {
    u'要', u'一', u'一切', u'去年', u'中', u'第二', u'再', u'而',
    u'怎麼', u'且', u'外', u'尤其', u'於是', u'些', u'真的', u'依',
    u'必須', u'後來', u'起來', u'那', u'錢', u'所以', u'被', u'未',
    u'然而', u'本身', u'一般', u'到', u'們', u'你們', u'段', u'場',
    u'其他', u'已', u'此外', u'常', u'隻', u'沒有', u'呀', u'部份',
    u'應', u'半', u'像', u'就', u'針對', u'不要', u'四', u'同時',
    u'例如', u'他', u'人', u'句', u'她', u'給', u'啦', u'經', u'歲',
    u'年', u'當', u'但是', u'七', u'如', u'還', u'第一', u'十分',
    u'下', u'二', u'最後', u'第三', u'因為', u'常常', u'能夠', u'別人',
    u'一些', u'對於', u'從', u'非常', u'還是', u'連', u'不但', u'目前',
    u'因此', u'吧', u'至於', u'除了', u'您', u'路', u'主要', u'是',
    u'之後', u'之間', u'誰', u'張', u'家', u'部分', u'只要', u'為什麼',
    u'十', u'可是', u'不', u'及', u'自己', u'或者', u'則', u'已經',
    u'等', u'才', u'之', u'其', u'位', u'其中', u'完全', u'嗎', u'最近',
    u'著', u'而且', u'多少', u'也', u'我們', u'以後', u'並且', u'不會',
    u'另', u'兩', u'都', u'可以', u'自', u'六', u'或是', u'可能', u'過去',
    u'極', u'今天', u'一直', u'用', u'似乎', u'書', u'好像', u'幾', u'好',
    u'身', u'與', u'的話', u'大家', u'它', u'太', u'約', u'其實', u'分',
    u'月', u'上', u'會', u'本', u'他們', u'間', u'只是', u'得', u'或',
    u'不能', u'者', u'條', u'共同', u'的', u'多', u'透過', u'事', u'時',
    u'將', u'另外', u'並', u'在', u'來', u'各', u'自我', u'種', u'地',
    u'妳', u'字', u'去', u'個', u'便', u'甚至', u'心', u'把', u'雖然',
    u'因', u'手', u'先', u'塊', u'仍', u'水', u'每', u'和', u'現在',
    u'如何', u'昨天', u'受', u'最', u'有的', u'跟', u'類', u'裡', u'你',
    u'呢', u'以', u'此', u'任何', u'以上', u'若', u'應該', u'嗯', u'由',
    u'至', u'該', u'一起', u'整', u'未來', u'我', u'能', u'比較', u'五',
    u'很多', u'往', u'較', u'只', u'項', u'所', u'三', u'很', u'但', u'邊',
    u'名', u'後', u'全', u'向', u'如果', u'這些', u'沒', u'可', u'了',
    u'那麼', u'一定', u'相當', u'真', u'亦', u'今年', u'不過', u'然後',
    u'那些', u'根據', u'這麼', u'幾乎', u'無法', u'為', u'於', u'更',
    u'這', u'次', u'元', u'啊', u'均', u'又', u'所有', u'前', u'當時',
    u'某', u'許多', u'比', u'當然', u'雖', u'過', u'卻', u'點', u'正',
    u'即使', u'內', u'這裡', u'是否', u'對', u'以及', u'由於', u'話',
    u'天', u'即', u'什麼', u'起', u'件', u'民國', u'曾',
}

chunk_nr = RegexpChunkParser([ChunkRule('<NR>+', 'nr')])
chunk_nt = RegexpChunkParser([ChunkRule('<NT>+', 'nt')])
chunk_nncc = RegexpChunkParser([ChunkRule('<NN|NR><NN|NR|CC|VV|JJ>*<NN|NR>', 'nncc')])


def chunk_parse(parser, text):
    nplist = []
    parsed_text = parser.parse(tagstr2tree(text))
    for node in parsed_text.productions():
        if str(node.lhs()) != 'NP':
            continue
        temp = []
        for w in node.rhs():
            try:
                w[0].encode('ascii')
                temp.append(' ')
                temp.append(w[0])
                temp.append(' ')
            except:
                temp.append(w[0])
        n = re.sub(' +', ' ', ''.join(temp)).strip()
        if n not in freq_words:
            nplist.append(n)
    return nplist


def np_chunking(sents):
    '''Given a list of words and a corresponding list of pos-tags,
    this function return a list of noun phrase chunks.'''
    if isinstance(sents, basestring):
        sents = [sents]

    nrlist = []
    nplist = []
    for sent in sents:
        text = ' '.join([w + '/' + t for w, t in zh.tw_postag(sent)
                         if w not in u'[]<>/{}'])
        nrlist.extend(chunk_parse(chunk_nr, text))
        nplist.extend(chunk_parse(chunk_nt, text))
        nplist.extend(chunk_parse(chunk_nncc, text))
    text = u'。'.join(sents)
    nplist.extend([w for w in re.findall(u'「(.+?)」', text) if len(w) < 7])
    nplist.extend([w for w in re.findall(u'『(.+?)』', text) if len(w) < 7])
    nplist.extend([w for w in re.findall(u'《(.+?)》', text) if len(w) < 7])
    nplist.extend([w for w in re.findall(u'”(.+?)”', text) if len(w) < 7])
    nplist.extend([w for w in re.findall(u'[^A-Za-z]([a-z]+)[^A-Za-z]', text)])
    nrlist.extend([w for w in re.findall(u'[^A-Za-z]([A-Z][A-Za-z]+)[^A-Za-z]', text)])
    nplist.extend(nrlist)
    nplist = list(set([n.strip() for n in nplist if len(n.strip()) > 1]))
    nrlist = list(set([n.strip() for n in nrlist if len(n.strip()) > 1]))
    # for i in range(len(nplist))[::-1]:
    #     if any([n.find(nplist[i]) >= 0 and n != nplist[i] for n in nplist]):
    #         nplist.pop(i)
    # for i in range(len(nrlist))[::-1]:
    #     if any([n.find(nrlist[i]) >= 0 and n != nrlist[i] for n in nrlist]):
    #         nrlist.pop(i)
    return list(set(nplist)), set(nrlist)


def find_keywords(text, select=None):
    if select is None:
        select = np.ones(len(text.sents), dtype=bool)
    np_list, nr_list = np_chunking(text.sents[select])
    article_vector = np.sum(text.vector[select], axis=0)
    keyword_vector = w2v.sentvec(np_list)
    sim = similarity(keyword_vector, article_vector)
    sim[np.isnan(sim)] = 1
    for i in range(len(sim)):
        sim[i] *= (1.10 ** (np_list[i] in nr_list))
        sim[i] *= (1.30 ** min(text.tidy_text.count(np_list[i]), 3))
        sim[i] *= (0.80 ** max(3 - zhlen(np_list[i]), 0))
        sim[i] *= (0.90 ** max(zhlen(np_list[i]) - 6, 0))
        sim[i] *= (1.10 ** all(ord(c) < 128 for c in np_list[i]))
    return np.array(np_list)[np.argsort(sim)[::-1]]
