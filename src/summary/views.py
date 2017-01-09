# -*- coding: utf-8 -*-
from django.http import JsonResponse
from datetime import datetime as dt
import numpy as np
from summarizer import Article, ParagraphSet, Sentence, summary_text


def handle(text, title):
        if len(title) > 10:
            article = Article(text, title=title)
        else:
            article = Article(text)
        para_set = article[0]
        keywords = para_set.keywords
        sents = [para_set[0]]  # always choose first sentence

        if len(article) == 1:
            sim = para_set.mostsim[:5].tolist()
            first_summary = 5
        elif len(para_set) >= 2 and para_set.title is None:
            sim = para_set.mostsim[:2].tolist()
        else:
            sim = para_set.mostsim[:1].tolist()

        if sim.count(0):
            sim.remove(0)
        else:
            sim.pop(-1)
        if len(sim):
            sents.extend(para_set.sents[np.sort(sim)])
        result = []

        s1, s2 = sents[0].seg, sents[0].seg
        if para_set.title is not None:
            s1 = para_set.title.seg + '<br>' + s1
            s2 = para_set.title.seg + '<br>' + s2
        result.append((s1, s2, "_".join(keywords)))
        for sent in sents[1:]:
            # result.append((sent.seg, sent.short.seg, "_".join(keywords)))
            result.append((sent.seg, sent.seg, "_".join(keywords)))

        if len(article) == 1:  # number of paragraphs is 1
            return result, article.lang

        for para_set in article[1:]:
            if para_set.title is not None:
                sent = para_set.sents[0]
            else:
                sent = para_set.summary[0]
            keywords = para_set.keywords
            # s1, s2 = sent.seg, sent.short.seg
            s1, s2 = sent.seg, sent.seg
            if para_set.title is not None:
                s1 = para_set.title.seg + '<br>' + s1
                s2 = para_set.title.seg + '<br>' + s2
            result.append((s1, s2, "_".join(keywords)))
        return result, article.lang


def parse_api(request):
    summary, lang = handle(request.POST['text'], request.POST['title'])
    ret = {
        'lang': lang,
        'short': summary,
    }
    return JsonResponse(ret)
