# -*- coding: utf-8 -*-
from django.http import JsonResponse
from zh_tree import ChineseTree
from zh_summary import summary_text, shorten_sents


def parse_api(request, num='0'):
    algorithm = int(num)
    raw_text = request.POST['text']
    summary = summary_text(raw_text, 5, algorithm)
    shorten = shorten_sents(summary)
    mergtree = ChineseTree(summary[0], name_with_pos=True)
    origtree = ChineseTree(summary[0], merging=False, name_with_pos=True)
    ret = {
        'raw': unicode(origtree),
        'short': zip(summary.tolist(), shorten.tolist()),
        'tree_orig': origtree.tree,
        'tree': mergtree.tree,
    }
    return JsonResponse(ret)
