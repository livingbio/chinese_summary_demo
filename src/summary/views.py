# -*- coding: utf-8 -*-
from django.http import JsonResponse
from datetime import datetime as dt
import numpy as np
from summarizer import Article, ParagraphSet, Sentence, summary_text
from summarizer.lib_detectlang import detect_lang


def parse_api(request):
    lang = detect_lang(request.POST['text'])
    summary = summary_text[lang[:2]](request.POST['text'], request.POST['title'])
    ret = {
        'short': summary,
    }
    return JsonResponse(ret)
