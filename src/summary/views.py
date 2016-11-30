# -*- coding: utf-8 -*-
from django.http import JsonResponse
from datetime import datetime as dt
import numpy as np
from summarizer import Article, ParagraphSet, Sentence, summary_text


def handle(text):
    article = Article(text)
    first_summary = 5 if len(article) == 1 else 2

    para_set = article[0]
    keywords = para_set.keywords
    sents = [para_set[0]]  # always choose first sentence
    sim = para_set.mostsim[:first_summary].tolist()
    if sim.count(0):
        sim.remove(0)
    else:
        sim.pop(-1)
    sents.extend(para_set.sents[np.sort(sim)])
    result = []
    for sent in sents:
        result.append((sent.seg, sent.short.seg, "_".join(keywords)))

    if len(article) == 1:  # number of paragraphs is 1
        return result

    for para_set in article[1:]:
        sent = para_set.summary[0]
        keywords = para_set.keywords
        if para_set.title is not None:
            s1 = para_set.title.seg + '<br>' + sent.seg
            s2 = para_set.title.seg + '<br>' + sent.short.seg
        else:
            s1, s2 = sent.seg, sent.short.seg
        result.append((s1, s2, "_".join(keywords)))
    return result


def parse_api(request):
    summary = handle(request.POST['text'])
    ret = {
        'short': summary,
    }
    return JsonResponse(ret)
