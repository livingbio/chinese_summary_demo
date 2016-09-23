# -*- coding: utf-8 -*-
from scipy.spatial.distance import cosine, cdist
from sklearn.cluster import AgglomerativeClustering
import numpy as np


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
    scores = similarity(sents_vector, article_vector) * score_reward
    clus_score = [np.mean(scores[cluster == c]) for c in range(n_dummy)]
    select_cluster = np.argsort(clus_score)[::-1][:n_summary]

    summary = np.zeros(n_summary, dtype=int)
    for i, c in enumerate(select_cluster):
        centroid = np.mean(sents_vector[cluster == 5], axis=0)
        score_inclus = similarity(sents_vector[cluster == 5], centroid)
        summary[i] = np.where(cluster == c)[0][np.argmax(score_inclus)]
    return np.sort(summary)


def maximal_summarizer(sents_vector, article_vector, n_summary, score_reward):
    '''Given a list of sentences, find `n_summary` sentences to summarize the text.
    Args:
        sentences (:obj:`list` of :obj:`str`): a list of sentences.
        n_summary (str): the number of sentences to be put into summary.
    Returns:
        :obj:`np.array` of :obj:`str`: a list of summary sentences.
    '''
    scores = similarity(sents_vector, article_vector) * score_reward
    best = np.argsort(scores)[-1:-(n_summary + 1):-1]
    return np.sort(best)
