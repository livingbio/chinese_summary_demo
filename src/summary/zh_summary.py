# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from libvec import sentence_vector, similarity, tidify, find_oov
from libparse import DependencyTree
import re

summary_data = {}


def to_half_word(text):
    '''Transfer double-width character to single-width character.'''
    return ''.join([chr(ord(ch) - 0xff00 + 0x20)
                    if ord(ch) >= 0xff01 and ord(ch) <= 0xff5e else ch
                    for ch in text])


def sent_tokenize(paragraph):
    text = re.findall(u'[^。！？；]+', paragraph)
    punc = re.findall(u'[。！？；]', paragraph)
    return [t + p for t, p in zip(text, punc)]


def split_sentence(paragraph):
    sum_sents = sent_tokenize(tidify(paragraph))
    disp_sents = [s.replace(' ', '') for s in sum_sents]
    return np.array(sum_sents), np.array(disp_sents)


def dynaprog_summarizer(sents, article_vec):
    '''Given a list of sentences, find some sentences to represent the text.
    Args:
        sentences (:obj:`list` of :obj:`str`): a list of sentences.
    Returns:
        :obj:`np.array` of :obj:`str`: a list of summary sentences.
    '''
    score_map = []
    best = None
    for i in range(len(sents)):
        score_map.append([])
        for j in range(i + 1):
            print("map[%d][%d]" % (i, j))
            score_map[i].append({})
            if j == 0:
                score_map[i][j]["score"] = similarity(article_vec, sents[i])
                score_map[i][j]["selected"] = [i]
                score_map[i][j]["vec"] = sents[i]
            else:
                score_map[i][j]["score"] = None
                for k in range(j - 1, i):
                    if k >= len(score_map):
                        break
                    if (j - 1) >= len(score_map[k]) or score_map[k][j - 1] is None:
                        continue

                    print("trying map[%d][%d]" % (k, j - 1))
                    vec = score_map[k][j - 1]["vec"] + sents[i]
                    s = similarity(article_vec, vec)
                    if (score_map[i][j]["score"] is None) or s > score_map[i][j]["score"]:
                        score_map[i][j]["score"] = s
                        score_map[i][j]["selected"] = list(score_map[k][j - 1]["selected"])
                        score_map[i][j]["selected"].append(i)
                        score_map[i][j]["vec"] = vec
            print("score[%d][%d] = %g, selected: %r" % (i, j, score_map[i][j]["score"], score_map[i][j]["selected"]))
            if (best is None) or score_map[i][j]["score"] > best["score"]:
                best = score_map[i][j]
            else:
                break
    if best:
        print(best["selected"])
        print(best["score"])
        return np.array(best["selected"])
    else:
        None


def cluster_summarizer(sents_vector, article_vector, n_summary, score_reward):
    '''Given a list of sentences, find `n_summary` sentences to summarize the text.
    Args:
        sentences (:obj:`list` of :obj:`str`): a list of sentences.
        n_summary (str): the number of sentences to be put into summary.
    Returns:
        :obj:`np.array` of :obj:`str`: a list of summary sentences.
    '''
    n_dummy = min(len(sents_vector), int(n_summary * 2))
    algo = AgglomerativeClustering(n_clusters=n_dummy)
    cluster = algo.fit_predict(sents_vector)
    scores = similarity(sents_vector, article_vector)
    for i in range(len(scores)):
        scores[i] *= score_reward[i]
    clus_score = [np.mean(scores[cluster == c]) for c in range(n_dummy)]
    select_cluster = np.argsort(clus_score)[::-1][:n_summary]

    summary = np.zeros(n_summary, dtype=int)
    for i, c in enumerate(select_cluster):
        summary[i] = np.where(cluster == c)[0][np.argmax(scores[cluster == c])]
    return np.sort(summary)


def maximal_summarizer(sents_vector, article_vector, n_summary, score_reward):
    '''Given a list of sentences, find `n_summary` sentences to summarize the text.
    Args:
        sentences (:obj:`list` of :obj:`str`): a list of sentences.
        n_summary (str): the number of sentences to be put into summary.
    Returns:
        :obj:`np.array` of :obj:`str`: a list of summary sentences.
    '''
    scores = similarity(sents_vector, article_vector)
    for i in range(len(scores)):
        scores[i] *= score_reward[i]
    best = np.argsort(scores)[-1:-(n_summary + 1):-1]
    return np.sort(best)


def find_special_nouns(raw_text):
    text = tidify(raw_text)
    words = np.array(text.split())
    proper_noun = set()
    for w in find_oov(words):
        cnt = raw_text.count(w)
        cntlower = raw_text.count(w.lower())
        # print 'OOV', w, cnt, cntlower, cnt > 4, (cnt - cntlower) > 2
        if w[0].isupper() and cnt > 4 and (cnt - cntlower) > 2:
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


def adjust_by_len(score, sentences, limit=60, decay=0.998):
    for i, s in enumerate(sentences):
        if len(s) > limit:
            panelty = decay ** (len(s) - limit)
        elif len(s) < limit / 2:
            panelty = decay ** (limit / 2 - len(s))
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
    raw_text = raw_text.replace('\n\n', ' .\n\n')
    summary_data['special_nouns'] = find_special_nouns(raw_text)
    print '>>> Proper Nouns', summary_data['special_nouns']

    sum_sents, disp_sents = split_sentence(raw_text)
    # score_reward = adjust_by_nouns([1.0] * len(disp_sents), disp_sents, growth=1.01)
    score_reward = [1.0] * len(disp_sents)
    sents_vector = sentence_vector(sum_sents)
    article_vector = np.sum(sents_vector, axis=0)
    summary_data['article_vector'] = article_vector

    if algorithm == 0:
        index = dynaprog_summarizer(sents_vector, article_vector)
    elif algorithm == 1:
        if len(disp_sents) <= n_summary:
            index = np.arange(len(disp_sents))
        else:
            index = cluster_summarizer(sents_vector, article_vector, n_summary, score_reward)
    elif algorithm == 2:
        index = maximal_summarizer(sents_vector, article_vector, n_summary, score_reward)
    return np.array(disp_sents[index])


def shorten_sents(summary):
    shorten = []
    for s in summary:
        if len(s) < 60:
            shorten.append([(1.0, s)])
            continue
        deptree = DependencyTree(tidify(s))
        chunks = deptree.chunking()
        score = similarity(sentence_vector(chunks), summary_data['article_vector'])
        score = adjust_by_nouns(score, chunks)
        score = adjust_by_len(score, chunks)
        shorten.append(zip(score, chunks))
    return np.array(shorten)
