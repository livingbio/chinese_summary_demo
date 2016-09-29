# -*- coding: utf-8 -*-
from django.http import JsonResponse
from zh_tree import ChineseTree
from zh_summary import summary_text, shorten_sents
from datetime import datetime as dt


def parse_api(request, num='0'):
    algorithm = int(num)
    raw_text = request.POST['text']
    start = dt.now()
    summary = summary_text(raw_text, 5, algorithm)
    print 'summary', dt.now() - start
    start = dt.now()
    shorten = shorten_sents(summary)
    print 'shorten', dt.now() - start
    start = dt.now()
    mergtree = ChineseTree(summary[0], name_with_pos=True)
    origtree = ChineseTree(summary[0], merging=False, name_with_pos=True)
    print 'tree', dt.now() - start
    ret = {
        'raw': unicode(origtree),
        'short': zip(summary.tolist(), shorten.tolist()),
        'tree_orig': origtree.tree,
        'tree': mergtree.tree,
    }
    return JsonResponse(ret)
