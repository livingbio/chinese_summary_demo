# -*- coding: utf-8 -*-
from django.http import JsonResponse
from zh_tree import ChineseTree
from zh_summary import summary_text, shorten_sents
from datetime import datetime as dt


def parse_api(request, num='0'):
    raw_text = request.POST['text']
    start = dt.now()
    summary = summary_text(raw_text, algorithm=int(num), shorten=False)
    shorten = summary_text(raw_text, algorithm=int(num))
    print 'summary', dt.now() - start
    if request.POST['tree'] == 'true':
        mergtree = ChineseTree(summary[0], name_with_pos=True)
        tree_merg = mergtree.tree.tree
        origtree = ChineseTree(summary[0], merging=False, name_with_pos=True)
        tree_orig = origtree.tree.tree
    else:
        origtree = ''
        tree_merg = []
        tree_orig = []
    ret = {
        'raw': unicode(origtree),
        'short': zip(summary.tolist(), shorten.tolist()),
        'tree_orig': tree_orig,
        'tree': tree_merg,
    }
    return JsonResponse(ret)
