# -*- coding: utf-8 -*-
from django.http import JsonResponse
import re
import os
import json
from libparse import DependencyTree
from libvec import tidify
from libsummary import summary_text, shorten_sents
from subprocess import Popen
import urllib2


def parse_api(request, num='0'):
    algorithm = int(num)
    raw_text = request.POST['text']
    summary = summary_text(raw_text, 5, algorithm)
    shorten = shorten_sents(summary)
    mergtree = DependencyTree(tidify(summary[0]), name_with_pos=True)
    origtree = DependencyTree(tidify(summary[0]), merging=False, name_with_pos=True)
    ret = {
        'raw': unicode(origtree),
        'short': zip(summary.tolist(), shorten.tolist()),
        'tree_orig': origtree.tree,
        'tree': mergtree.tree,
    }
    return JsonResponse(ret)
