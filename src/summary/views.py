# -*- coding: utf-8 -*-
from django.http import JsonResponse
from datetime import datetime as dt
import numpy as np
from summarizer import Article, ParagraphSet, Sentence, summary_text

from summarizer.lib_keyword import chunk_parse
from nltk import RegexpChunkParser
from nltk.chunk.regexp import ChunkRule
from collections import Counter


def find_person(a):
    person = RegexpChunkParser([ChunkRule(u'<NP>+', 'noun')])
    person2 = RegexpChunkParser([ChunkRule(u'<NP>+<NN>?<NX>', 'noun')])
    person3 = RegexpChunkParser([ChunkRule(u'<SP>+', 'noun')])

    name_set = set()
    cnt = Counter()
    for s in a.sents:
        names = chunk_parse(person, s.tagged) + chunk_parse(person2, s.tagged)
        name_set.update(names)
        cnt.update(names)
        names = [n.replace('_', ' ') for n in chunk_parse(person3, s.tagged) if n[0].isupper()]
        name_set.update(names)
        cnt.update(names)

    name_set = list(name_set)
    name_parent = list()
    for np in name_set:
        par = [(len(n), n) for n in name_set if n.count(np) > 0 and len(n) > len(np)]
        if not par:
            name_parent.append(np)
        else:
            name_parent.append(sorted(par)[-1][1])

    for np, c in cnt.most_common():
        i = name_set.index(np)
        if name_parent[i] != name_set[i]:
            cnt[name_parent[i]] += c
    return cnt.most_common()


def handle(text):
    article = Article(text)
    if len(article) == 1:
        first_summary = 5
    elif len(article[0]) >= 2:
        first_summary = 2
    else:
        first_summary = 1

    para_set = article[0]
    keywords = para_set.keywords.tolist()
    if article.lang == 'ja':
        keywords.insert(0, find_person(article)[0][0])
    sents = [para_set[0]]  # always choose first sentence
    sim = para_set.mostsim[:first_summary].tolist()
    if sim.count(0):
        sim.remove(0)
    else:
        sim.pop(-1)
    if len(sim):
        sents.extend(para_set.sents[np.sort(sim)])
    result = []
    for sent in sents:
        # result.append((sent.seg, sent.short.seg, "_".join(keywords)))
        result.append((sent.seg, sent.seg, "_".join(keywords)))

    if len(article) == 1:  # number of paragraphs is 1
        return result, article.lang

    for para_set in article[1:]:
        sent = para_set.summary[0]
        keywords = para_set.keywords.tolist()
        if article.lang == 'ja':
            keywords.insert(0, find_person(article)[0][0])
        # s1, s2 = sent.seg, sent.short.seg
        s1, s2 = sent.seg, sent.seg
        if para_set.title is not None:
            s1 = para_set.title.seg + '<br>' + s1
            s2 = para_set.title.seg + '<br>' + s2
        result.append((s1, s2, "_".join(keywords)))
    return result, article.lang


def parse_api(request):
    summary, lang = handle(request.POST['text'])
    ret = {
        'lang': lang,
        'short': summary,
    }
    return JsonResponse(ret)
