# -*- coding: utf-8 -*-
import numpy as np
import re
from zh_lib import w2v, zh, zhlen, adjust_by_len, adjust_by_nouns
from zh_tree import ChineseTree
from zh_grouper import TextGrouper
from zh_keyword import find_keywords
from summarizer import similarity, maximal_summarizer

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
    seg = [w.replace('_', ' ') for w in (zh.tw_segment(string) or [])]
    return '_'.join(seg)


def chunking_sent(sentence, forceFirstSubSent=False):
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
            shorten.append(sorted([ch for ch in zip(score, chunks) if not np.isnan(ch[0])]))
        else:
            shorten.append([(1.0, newline_hint(s))])
    return np.array(shorten)


def summary_text(raw_text, n_summary=5, algorithm=0, shorten=True):
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
    assert algorithm == 0  # we don't use this parameter now

    text = TextGrouper(raw_text)
    # special_nouns = find_special_nouns(text.tidy_text)
    # print '>>> Proper Nouns', '/'.join(special_nouns)

    sents = text.sents
    sents_vector = text.vector
    if text.predn <= 1:  # no text group
        article_vector = np.sum(sents_vector, axis=0)
        index = maximal_summarizer(sents_vector, article_vector, n_summary)
        kw = '_'.join(find_keywords(text)[:5])
        keywords = [kw] * len(index)
    else:
        groups = text.group()
        group_size = [max(1, n_summary // len(groups))] * len(groups)
        if sum(group_size) < n_summary:
            group_size[0] += n_summary - sum(group_size)
        group_vector = [np.sum(sents_vector[g], axis=0) for g in groups]
        index = []
        keywords = []
        used_keywords = set()
        for gvec, gid, gsize in zip(group_vector, groups, group_size):
            sel = maximal_summarizer(sents_vector[gid], gvec, gsize)
            index.extend(gid[sel])
            kw = '_'.join([k for k in find_keywords(text, gid)
                           if k not in used_keywords][:2])
            used_keywords |= set(kw.split('_'))
            for i in range(gsize):
                keywords.append(kw)
        if text.sent_numbers is not None:
            tmp_kw = zip(index, keywords)
            tmp_kw.extend(zip(text.sent_numbers, [''] * len(text.sent_numbers)))
            index.extend(text.sent_numbers.tolist())
            index = sorted(index)
            keywords = [kw for i, kw in sorted(tmp_kw)]

    if shorten:
        short = shorten_sents(sents[index])
    else:
        short = [[(1.0, s)] for s in sents[index]]
    summary = []
    last_sum = None
    for i, sent, short_sent, kw in zip(index, sents[index], short, keywords):
        if text.sent_numbers is not None and i in text.sent_numbers:
            last_sum = newline_hint(sent)
            last_short = short_sent[-1][1]
        else:
            if last_sum is not None:
                sent = last_sum + '_<br>_' + sent
                short = last_short + '_<br>_' + short_sent[-1][1]
                summary.append((newline_hint(sent), short, kw))
                last_sum = None
            else:
                summary.append((newline_hint(sent), short_sent[-1][1], kw))
    return np.array(summary)
