# -*- coding: utf-8 -*-
from gliacloud_api_client import Parser
import re


rel_should_merge = {
    'acl': 'clausal modifier of noun',  # 一個範圍(之內) (出來)迎接
    # 'acl:relcl': 'acl:relcl',  # 昂貴(的)設備 美國(的)首都
    # 'advcl': 'adverbial clause modifier',  # (一般來說)，跑車較貴
    # 'advmod': 'adverbial modifier',  # (大)多數 (多半)放在桌上
    'amod': 'adjectival modifier',  # (昂貴)的設備
    'appos': 'appositional modifier',  # 化外之地，(文明地區以外的地方)，
    'asp': 'asp',
    'assmod': 'assmod',
    'aux': 'auxiliary',  # 你(可能)錯了 我(會)幫你
    'aux:caus': 'aux:caus',  # 我(把)家裡整理的很乾淨 他(將)酒一飲而盡
    'auxpass': 'passive auxiliary',  # 他(被)擊中了
    'case': 'case marking',  # 玩家將(和)敵人作戰 (從)東門走到西門
    'case:aspect': 'case:aspect',  # 他對此表示(了)不滿
    'case:dec': 'case:dec',  # 公司(的)財產 迪士尼(的)公主
    'case:pref': 'case:pref',  # (總)面積 世界三(大)男高音
    'case:suff': 'case:suff',  # (愛斯基摩)人 (電視)機 (加長)型禮車
    # 'cc': 'coordinating conjunction',  # 我(和)你 法魯克(與)拳四郎的對決
    'ccomp': 'clausal complement',  # 這台車歸(私人擁有) 我說(你可能錯了)
    'clf': 'clf',
    'compound': 'compound',
    # 'conj': 'conjunct',  # 愛斯基摩人和(維京人)
    'cop': 'copula',  # 這台車(則是)公司的財產
    'csubj': 'clausal subject',  # (這條線代表的)是100米
    'csubjpass': 'clausal passive subject',  # (燒荒肥田)曾被廣泛應用
    'dep': 'unspecified dependency',  # 人口12萬人((2009年))
    'det': 'determiner',  # (這台)車 (公司的)財產 (我和你的)交往
    'discourse': 'discourse element',  # 我可能猜錯(了) 這是他的責任(呀)
    # 'dislocated': 'dislocated elements',  # 這部分(我都看過) 會議(旨在發展經濟)
    'dobj': 'direct object',  # 升為副(教授) 購買(設備) 前往(東京)
    'dvpmod': 'dvpmod',
    # 'expl': 'expletive',
    'foreign': 'foreign words',  # 表面溫度(10000K) 由吉布斯((Gibbs))設計
    'goeswith': 'goes with',
    'iobj': 'indirect object',  # 東區併入(西區) 把梨子讓給(弟弟)
    'list': 'list',
    'loc': 'loc',
    'mark': 'marker',  # 移動(時)要注意 (而)工廠則停止生產
    'mmod': 'mmod',
    'mwe': 'multi-word expression',
    'nn': 'nn',
    'nfincl': 'nfincl',
    'neg': 'negation modifier',  # (未)完工 (不)奇怪
    # 'nmod': 'nominal modifier',  # (網路)公司 (美)元
    # 'nmod:tmod': 'nmod:tmod',  # (昨天上午)，他出來走走 英語(長期)是官方語言
    # 'nsubj': 'nominal subject',  # (愛斯基摩人和維京人)定居在此
    'nsubjpass': 'passive nominal subject',  # (系統)被破壞
    'nummod': 'numeric modifier',  # (四百五十萬)美元
    'ordmod': 'ordmod',
    # 'prep': 'prep',
    # 'parataxis': 'parataxis',
    # 'punct': 'punctuation',
    'prtmod': 'prtmod',
    'range': 'range',
    'rcomp': 'rcomp',
    # 'remnant': 'remnant in ellipsis',  # 北京城有七門，(南門三門)，(東西各一門)
    # 'reparandum': 'overridden disfluency',
    'relcl': 'relcl',
    # 'root': 'root',
    # 'tmod': 'tmod',
    'vocative': 'vocative',
    'xcomp': 'open clausal complement',  # 被認為(是違禁品) 開始(變得頻繁)
}

valid_ending = {'root', 'punct', 'etc'}

rule_connect = {
    'prep', 'conj', 'cc', 'nsubj', 'dobj', 'dep', 'punct',
    'advmod', 'mmod', 'ccomp', 'relcl', 'cop',
}

must_have_child = [
    {
        'punct', 'loc', 'advmod', 'case', 'clf', 'dobj',
        'nsubj', 'ba', 'relcl', 'assmod', 'neg', 'nn', 'cop',
        'amod', 'tmod', 'ccomp',
    },
    {
        'punct', 'loc', 'dep', 'advmod', 'case', 'clf', 'dobj',
        'nsubj', 'ba', 'relcl', 'assmod', 'neg', 'nn', 'cop',
        'amod', 'tmod', 'ccomp',
    },
    {
        'punct', 'loc', 'dep', 'advmod', 'case', 'clf', 'dobj',
        'nsubj', 'ba', 'relcl', 'assmod', 'neg', 'nn', 'cop',
        'amod', 'tmod', 'conj', 'cc', 'ccomp',
    },
    {
        'punct', 'loc', 'dep', 'advmod', 'case', 'clf', 'dobj',
        'nsubj', 'ba', 'relcl', 'assmod', 'neg', 'nn', 'cop',
        'amod', 'tmod', 'conj', 'cc', 'ccomp', 'prep',
    }
]


class ParseNode(object):
    def __init__(self):
        self.id = -1
        self.parent = None
        self.pos = 'root'
        self.rel = 'root'
        self.children = list()
        self.mergelist = list()
        self.merged = False
        self.size = 1

    def __unicode__(self):
        s = u'[{}] {}: POS={}, Parent={}, Relation={}, Merge={}'
        return s.format(self.id, self.name, self.pos, self.parent.id, self.rel, self.mergelist)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def dict(self, name_with_pos=False):
        if name_with_pos:
            name = u'{}({})'.format(self.name, self.pos)
        else:
            name = self.name
        d = {
            'id': self.id,
            'name': name,
            'pos': self.pos,
            'rel': self.rel,
            'parent': self.parent.id,
            'children': list(self.children),
            'mergelist': list(self.mergelist),
            'merged': self.merged,
            'size': self.size,
        }
        return d


def bfs(tree):
    retbfs_queue = []
    bfs_queue = [tree]
    tree['depth'] = 1
    while bfs_queue:
        tree = bfs_queue.pop(0)
        retbfs_queue.append(tree)
        for child in tree['children']:
            child['depth'] = tree['depth'] + 1
            bfs_queue.append(child)
    return retbfs_queue


def dfs(tree):
    dfs_queue = []
    dfs_stack = [tree]
    while dfs_stack:
        tree = dfs_stack.pop()
        dfs_queue.append(tree)
        for child in tree['children'][::-1]:
            dfs_stack.append(child)
    return dfs_queue


class TreeNode(object):
    def __init__(self, nodes, index, isNameWithPOS):
        self.isNameWithPOS = isNameWithPOS
        self.tree = self.create(nodes, index, 1)

    def create(self, nodes, index, depth):
        tree = nodes[index].dict(self.isNameWithPOS)
        tree['depth'] = depth
        tree['children'] = [self.create(nodes, ch, depth + 1)
                            for ch in tree['children'] if nodes[ch] is not None]
        tree['size'] += sum([ch['size'] for ch in tree['children']])
        return tree

    def roots(self):
        roots_id = []
        for tree in bfs(self.tree):
            if tree['rel'] == 'root':
                roots_id.append(tree['id'])
                tree['root'] = True
            elif tree['rel'] in ('conj', 'ccomp') and tree['size'] >= 20:
                roots_id.append(tree['id'])
                tree['root'] = True
            elif tree['rel'] == 'prep' and tree['size'] >= 20:
                roots_id.append(tree['parent'])
                tree['root'] = True
            else:
                tree['root'] = False
        roots = []
        roots_id = set(roots_id)
        for tree in bfs(self.tree):
            if tree['id'] in roots_id:
                roots.append(tree)
        return roots


class ChineseTree(object):
    def __init__(self, sentence, **kwargs):
        if 'merging' in kwargs:
            self.isMerging = kwargs['merging']
        else:
            self.isMerging = True
        if 'name_with_pos' in kwargs:
            self.isNameWithPOS = kwargs['name_with_pos']
        else:
            self.isNameWithPOS = False

        sentence = re.sub('([A-Za-z]) ([A-Za-z])', '\\1_\\2', sentence)
        sentence = re.sub('([A-Za-z]) ([A-Za-z])', '\\1_\\2', sentence)
        sentence = sentence.replace(' ', '')
        raw = Parser('zh').parse(sentence)[0]
        n_nodes = len(raw) + 1
        nodes = [ParseNode() for _ in range(n_nodes)]  # nodes[0] is dummy root
        self.puncts = []
        for n in raw:
            i = n['id']
            p = n['parent']
            if n['rel'] == 'punct' and n['parent'] == 0:
                p = i - 1
            nodes[i].id = n['id']
            nodes[i].name = n['name'].replace('_', ' ').replace(',', u'，')
            nodes[i].pos = n['pos']
            nodes[i].parent = nodes[p]
            nodes[i].rel = n['rel']
            if n['rel'] == 'punct':
                self.puncts.append(nodes[i])
            nodes[i].next = nodes[i + 1] if i < n_nodes - 1 else nodes[0]
            nodes[i].prev = nodes[i - 1] if i > 0 else nodes[-1]
            nodes[p].children.append(i)
        self.nodes = nodes
        # 解決標點符號位置的問題
        for n in self.puncts:
            p = n.next
            while p.parent and p.parent.id != n.prev.id:
                p = p.parent
            if p.rel != 'root':
                n.parent.children.remove(n.id)
                n.parent = n.prev
                n.parent.children.append(n.id)
                n.parent.children = sorted(n.parent.children)

        self.analyse_merge()  # prepare 'merged' and 'mergelist' attributes
        self.execute_merge()  # if self.isMerging, merge nodes by 'mergelist'
        self.root_index = nodes[0].children[0]
        self.tree = TreeNode(nodes, self.root_index, self.isNameWithPOS)
        self.tree.roots()

    def __unicode__(self):
        nonempty = [n for n in self.nodes[1:] if n]
        return u'\n'.join(map(unicode, nonempty))

    def __str__(self):
        return unicode(self).encode('utf-8')

    def analyse_merge(self):
        nodes = self.nodes
        for n in nodes[1:]:
            # 符合merge條件的relation
            if n.rel in rel_should_merge:
                n.parent.mergelist.append(n.id)
            elif n.rel == 'advmod' and n.prev.rel not in ('punct', ''):
                n.parent.mergelist.append(n.id)
            elif n.rel == 'prep' and n.pos == 'VERB':
                n.parent.mergelist.append(n.id)
            elif n.rel == 'nsubj' and len(n.children) == 0:
                n.parent.mergelist.append(n.id)
            elif n.rel == 'conj' and n.next.rel == 'cc' and len(n.parent.children) == 2:
                n.parent.mergelist.append(n.id)
                n.parent.mergelist.append(n.next.id)
            elif n.rel == 'cc' and n.next.rel == 'dep' and \
                len(n.parent.children) == 2 and len(n.next.children) == 0:
                n.parent.mergelist.append(n.id)
                n.parent.mergelist.append(n.next.id)

        for n in nodes[1:]:
            if not n.mergelist:  # 如果mergelist是空的
                continue
            cont_merge = []
            left_child = sorted([ch for ch in n.children if ch < n.id])[::-1]
            righ_child = sorted([ch for ch in n.children if ch > n.id])
            # 為了保持語法的正確，一定會merge連續的子節點
            # 如果node 11的子節點是[5, 9, 10, 12, 13]、mergelist=[9, 10, 13]
            # 則left_child=[10, 9, 5]、righ_child=[12, 13]
            # 原本9應該被merge，但因為10不能merge，所以9也連帶不能merge
            for ch in left_child:
                if ch in n.mergelist:
                    cont_merge.append(ch)
                else:
                    break
            for ch in righ_child:
                if ch in n.mergelist:
                    cont_merge.append(ch)
                else:
                    break
            n.mergelist = sorted(cont_merge)
            # 需要被merge的子節點，會有"merged"屬性
            for ch in n.mergelist:
                nodes[ch].merged = True

    def merge_single_node(self, n, mergelist):
        nodes = self.nodes
        # 名稱會依照children及本身的順序組合
        namelist = [(nodes[ch].id, nodes[ch].name) for ch in mergelist]
        namelist = sorted(namelist + [(n.id, n.name)])
        for i in range(1, len(namelist)):
            try:
                namelist[i - 1][1].encode('ascii')
                namelist[i][1].encode('ascii')
                namelist.append((namelist[i][0], ' '))
            except:
                pass
        namelist = sorted(namelist)
        n.name = '_'.join([name for _, name in namelist])
        n.size += sum([nodes[ch].size for ch in mergelist])
        # 收集subtree出現過的postag, relation以供後續分析
        for ch in mergelist:
            n.pos_list.extend(nodes[ch].pos_list)
            n.rel_list.extend(nodes[ch].rel_list)
        n.pos_list = sorted(n.pos_list)
        n.rel_list = sorted(n.rel_list)
        # 被merge的點就從nodes中消失了，本來的位置設為None
        for ch in mergelist:
            p = nodes[ch].parent
            p.children.remove(ch)
            p.children.extend(nodes[ch].children)
            for i in nodes[ch].children:
                nodes[i].parent = p
            nodes[ch] = None
        n.children = sorted(n.children)

    def recursive_merge(self, nodes, index):
        n = nodes[index]
        n.pos_list = [(n.id, n.pos)]
        n.rel_list = [(n.id, n.rel)]
        if n is None or not n.children:  # an empty node or a node without child
            return
        for ch in n.children:  # each child should be merged first
            self.recursive_merge(nodes, ch)
        if len(n.children) == 1 and nodes[n.children[0]].rel == 'punct':
            n.mergelist.append(n.children[0])
        if not n.mergelist and n.merged:  # n有children, 但沒有mergelist, 就不能被merge
            n.merged = False
            if n.id in n.parent.mergelist:
                n.parent.mergelist.remove(n.id)
            return
        mergelist = []
        for ch in sorted(filter(lambda x: x < n.id, n.children))[::-1]:
            if nodes[ch].merged:
                mergelist.append(ch)
            else:
                break
        for ch in sorted(filter(lambda x: x > n.id, n.children)):
            if nodes[ch].merged:
                mergelist.append(ch)
            else:
                break
        self.merge_single_node(n, mergelist)
        # 嘗試merge children sibling
        mergelist = []
        for ch in n.children:
            if len(nodes[ch].children) == 0 and nodes[ch].rel in rel_should_merge:
                mergelist.append(ch)
            elif len(mergelist) > 1:
                self.merge_single_node(nodes[mergelist[0]], mergelist[1:])
                mergelist = []
            else:
                mergelist = []

        # 如果任一子節點無法被merge，則就不該再往上merge，應該設自己merged=False
        if n.parent.children and any([not nodes[ch].merged for ch in n.children]):
            n.merged = False
            if n.id in n.parent.mergelist:
                n.parent.mergelist.remove(n.id)

    def execute_merge(self):
        if not self.isMerging:
            return
        self.recursive_merge(self.nodes, self.nodes[0].children[0])
        # 重新產生next, preve兩個指標
        nonempty = [i for i, n in enumerate(self.nodes) if n] + [0]
        for i, nid in zip(range(1, len(nonempty)), nonempty[1:-1]):
            self.nodes[nid].next = self.nodes[nonempty[i + 1]]
            self.nodes[nid].prev = self.nodes[nonempty[i - 1]]

    def validate_names(self, names):
        # 如果有連接詞'cc'，就一定要有連接的部分'conj'
        for i in range(len(names) - 1):
            if names[i][2] == 'cc' and names[i + 1][2] != 'conj':
                names[i] = (0, '', '')
            if names[i][2] == 'punct' and names[i + 1][2] == 'punct':
                names[i] = (0, '', '')
            if names[i][2] == 'punct' and names[i][1] == u'、' and names[i + 1][2] != 'conj':
                names[i] = (0, '', '')
        ids = set([i for i, _, _ in names])
        for n in self.nodes[1:]:
            if n and n.id not in ids and n.name == u'、' and n.next.id in ids and n.prev.id in ids:
                names.append((n.id, n.name, n.rel))
        names = sorted([n for n in names if n[0] > 0])
        while names and names[0][2] in ('punct', 'mark', 'advmod'):
            del names[0]
        while names and names[-1][2] in ('punct', 'mark', 'advmod'):
            del names[-1]
        if self.nodes[names[-1][0]].next.rel not in valid_ending:
            return [(0, '', '')]
        return names

    def generate_names(self, trees):
        names = []
        for t in trees:
            try:
                names[-1][1].encode('ascii')
                t['name'].encode('ascii')
                names.append((t['id'], ' ', ''))
            except:
                pass
            names.append((t['id'], t['name'], t['rel']))
        names = self.validate_names(sorted(names))
        return '_'.join([nn for _, nn, _ in names])

    def append_treenode(self, ids, trees, tree, child_rule):
        if tree['id'] in ids:
            return
        # print '    -add-', tree['id'], tree['rel'], tree['name']
        # print '        -ch-', ' '.join([ch['name'] for ch in tree['children']])
        trees.append(tree)
        ids.append(tree['id'])
        has_one_conj = False
        if len(tree['children']) == 2:
            a, b = tree['children']
            if a['rel'] == 'conj' and not a['children'] and b['rel'] == 'cc' and not b['children']:
                self.append_treenode(ids, trees, a, child_rule)
                self.append_treenode(ids, trees, b, child_rule)
                has_one_conj = True
        for ch in tree['children']:
            if ch['rel'] in child_rule:
                self.append_treenode(ids, trees, ch, child_rule)
            elif ch['rel'] == 'prep' and len(ch['children']) == 0:
                self.append_treenode(ids, trees, ch, child_rule)
            elif ch['rel'] == 'mark' and ch['pos'] == 'PRT' and len(ch['children']) == 0:
                self.append_treenode(ids, trees, ch, child_rule)
            elif not has_one_conj and ch['rel'] == 'conj' and ch['pos'] == 'VERB':
                has_one_conj = True
                self.append_treenode(ids, trees, ch, child_rule)

    def rule_chunking(self, root):
        chunks = list()
        for i, child_rule in enumerate(must_have_child):
            ids, trees = list(), list()
            # print '------- rule {} -------'.format(i + 1)
            self.append_treenode(ids, trees, root, child_rule)
            depth = 1
            for tree in bfs(root):
                if tree['parent'] not in ids or tree['id'] in ids:
                    continue
                n = self.nodes[tree['id']]
                if tree['depth'] > depth:
                    depth = tree['depth']
                    chunks.append(self.generate_names(trees))
                    # print '    -gen-', chunks[-1]
                if n.prev.rel == 'punct' and n.pos_list[0][1] in ('ADP', 'ADV') \
                    and n.prev.prev.pos_list[-1][1] != 'PRT':
                    chunks.append(self.generate_names(trees))
                    # print '    -gen-', chunks[-1]
                if tree['rel'] in rule_connect:
                    self.append_treenode(ids, trees, tree, child_rule)
            chunks.append(self.generate_names(trees))
        return chunks

    def chunking(self):
        roots = self.tree.roots()
        chunks = []
        for root in roots:
            chunks.extend(self.rule_chunking(root))
        chunks = [ch for ch in set(chunks) if ch]
        return chunks
