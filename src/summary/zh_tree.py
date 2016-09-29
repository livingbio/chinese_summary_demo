# -*- coding: utf-8 -*-
from parser_client import Parser
from zhconvert import conv2cn, ZHConvert
import re

try:
    zh = ZHConvert('http://localhost:9998/pos?wsdl', 'http://localhost:9999/seg?wsdl')
    zh.tw_postag(u'今天天氣真好')
except:
    zh = ZHConvert()

universal = {
    'AD': 'ADV', 'AS': 'PRT', 'BA': 'X', 'CC': 'CONJ', 'CD': 'NUM', 'CS': 'CONJ',
    'DEC': 'PRT', 'DEG': 'PRT', 'DER': 'PRT', 'DEV': 'PRT', 'DT': 'DET',
    'ETC': 'PRT', 'FW': 'X', 'IJ': 'X', 'JJ': 'ADJ', 'LB': 'X', 'LC': 'PRT',
    'M': 'NUM', 'MSP': 'PRT', 'NN': 'NOUN', 'NR': 'NOUN', 'NT': 'NOUN',
    'OD': 'NUM', 'ON': 'X', 'P': 'ADP', 'PN': 'PRON', 'PU': 'PUNCT', 'SB': 'X',
    'SP': 'PRT', 'VA': 'VERB', 'VC': 'VERB', 'VE': 'VERB', 'VV': 'VERB', 'X': 'X',
}


def tagged_sent_to_conll(tagged_sent):
    conll = []
    num = 1
    for word, postag in tagged_sent:
        x = [str(num), word, '_', universal[postag], postag, '_', '0', '_', '_', '_']
        conll.append('\t'.join(x))
        num += 1
    return '\n'.join(conll) + '\n'


rel_should_merge = {
    'acl': 'clausal modifier of noun',  # 一個範圍(之內) (出來)迎接
    # 'acl:relcl': 'acl:relcl',  # 昂貴(的)設備 美國(的)首都
    # 'advcl': 'adverbial clause modifier',  # (一般來說)，跑車較貴
    'advmod': 'adverbial modifier',  # (大)多數 (多半)放在桌上
    'amod': 'adjectival modifier',  # (昂貴)的設備
    'appos': 'appositional modifier',  # 化外之地，(文明地區以外的地方)，
    'aux': 'auxiliary',  # 你(可能)錯了 我(會)幫你
    'aux:caus': 'aux:caus',  # 我(把)家裡整理的很乾淨 他(將)酒一飲而盡
    'auxpass': 'passive auxiliary',  # 他(被)擊中了
    'case': 'case marking',  # 玩家將(和)敵人作戰 (從)東門走到西門
    'case:aspect': 'case:aspect',  # 他對此表示(了)不滿
    'case:dec': 'case:dec',  # 公司(的)財產 迪士尼(的)公主
    'case:pref': 'case:pref',  # (總)面積 世界三(大)男高音
    'case:suff': 'case:suff',  # (愛斯基摩)人 (電視)機 (加長)型禮車
    # 'cc': 'coordinating conjunction',  # 我(和)你 法魯克(與)拳四郎的對決
    # 'ccomp': 'clausal complement',  # 這台車歸(私人擁有) 我說(你可能錯了)
    'compound': 'compound',
    # 'conj': 'conjunct',  # 愛斯基摩人和(維京人)
    'cop': 'copula',  # 這台車(則是)公司的財產
    'csubj': 'clausal subject',  # (這條線代表的)是100米
    'csubjpass': 'clausal passive subject',  # (燒荒肥田)曾被廣泛應用
    # 'dep': 'unspecified dependency',  # 人口12萬人((2009年))
    'det': 'determiner',  # (這台)車 (公司的)財產 (我和你的)交往
    # 'discourse': 'discourse element',  # 我可能猜錯(了) 這是他的責任(呀)
    # 'dislocated': 'dislocated elements',  # 這部分(我都看過) 會議(旨在發展經濟)
    'dobj': 'direct object',  # 升為副(教授) 購買(設備) 前往(東京)
    # 'expl': 'expletive',
    'foreign': 'foreign words',  # 表面溫度(10000K) 由吉布斯((Gibbs))設計
    'goeswith': 'goes with',
    'iobj': 'indirect object',  # 東區併入(西區) 把梨子讓給(弟弟)
    'list': 'list',
    # 'mark': 'marker',  # 移動(時)要注意 (而)工廠則停止生產
    'mwe': 'multi-word expression',
    'name': 'name',
    'neg': 'negation modifier',  # (未)完工 (不)奇怪
    'nmod': 'nominal modifier',  # (網路)公司 (美)元
    # 'nmod:tmod': 'nmod:tmod',  # (昨天上午)，他出來走走 英語(長期)是官方語言
    'nsubj': 'nominal subject',  # (愛斯基摩人和維京人)定居在此
    'nsubjpass': 'passive nominal subject',  # (系統)被破壞
    'nummod': 'numeric modifier',  # (四百五十萬)美元
    # 'parataxis': 'parataxis',
    # 'punct': 'punctuation',
    # 'remnant': 'remnant in ellipsis',  # 北京城有七門，(南門三門)，(東西各一門)
    # 'reparandum': 'overridden disfluency',
    'root': 'root',
    'vocative': 'vocative',
    'xcomp': 'open clausal complement',  # 被認為(是違禁品) 開始(變得頻繁)
}

chunking_postag = {
    'VP': {'VERB'},
    'NP': {'NOUN', 'DET'},
}

must_connect_rel = {
    'ccomp', 'xcomp', 'dobj', 'iobj',
}


class TreeNode(object):
    def __init__(self):
        self.id = -1
        self.parent = None
        self.pos = ''
        self.rel = ''
        self.children = list()
        self.mergelist = list()
        self.merged = False

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
        }
        return d


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

        # tagged_sent = zh.cn_postag(conv2cn(sentence))
        # raw = Parser('zh').parse_raw(tagged_sent_to_conll(tagged_sent))[0]
        raw = Parser('zh').parse(conv2cn(sentence))[0]
        n_nodes = len(raw) + 1
        nodes = [TreeNode() for _ in range(n_nodes)]  # nodes[0] is dummy root
        offset = 0
        for n in raw:
            i = n['id']
            p = n['parent']
            nodes[i].id = n['id']
            nodes[i].name = sentence[offset:(offset + len(n['name']))]
            offset += len(n['name'])
            nodes[i].pos = n['pos']
            nodes[i].parent = nodes[p]
            nodes[i].rel = n['rel']
            nodes[i].next = nodes[i + 1] if i < n_nodes - 1 else nodes[0]
            nodes[i].prev = nodes[i - 1] if i > 0 else nodes[-1]
            nodes[p].children.append(i)
        self.nodes = nodes

        self.analyse_merge()  # prepare 'merged' and 'mergelist' attributes
        self.execute_merge()  # if self.isMerging, merge nodes by 'mergelist'
        # if self.isNameWithPOS == True, names of nodes contain pos-tag
        self.root_index = nodes[0].children[0]
        self.tree = self.create_tree(nodes, self.root_index)

    def __unicode__(self):
        nonempty = [n for n in self.nodes[1:] if n]
        return u'\n'.join(map(unicode, nonempty))

    def __str__(self):
        return unicode(self).encode('utf-8')

    def create_tree(self, nodes, index):
        tree = nodes[index].dict(self.isNameWithPOS)
        tree['children'] = [self.create_tree(nodes, ch)
                            for ch in tree['children'] if nodes[ch] is not None]
        return tree

    def analyse_merge(self):
        nodes = self.nodes
        for n in nodes[1:]:
            # 符合merge條件的relation
            if n.rel in rel_should_merge:
                n.parent.mergelist.append(n.id)
            elif n.rel == 'acl:relcl' and len(n.name) == 1:
                n.parent.mergelist.append(n.id)

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

    def recursive_merge(self, nodes, index):
        n = nodes[index]
        n.pos_list = [(n.id, n.pos)]
        n.rel_list = [(n.id, n.rel)]
        if n is None or not n.children:  # an empty node or a node without child
            return
        for ch in n.children:  # each child should be merged first
            self.recursive_merge(nodes, ch)
        if not n.mergelist and n.merged:
            n.merged = False
            if n.id in n.parent.mergelist:
                n.parent.mergelist.remove(n.id)
            return

        # 名稱會依照children及本身的順序組合
        namelist = [(nodes[ch].id, nodes[ch].name) for ch in n.mergelist]
        namelist = sorted(namelist + [(n.id, n.name)])
        n.name = ' '.join([name for _, name in namelist])
        # 收集subtree出現過的postag, relation以供後續分析
        for ch in n.mergelist:
            n.pos_list.extend(nodes[ch].pos_list)
        for ch in n.mergelist:
            n.rel_list.extend(nodes[ch].rel_list)
        n.pos_list = sorted(n.pos_list)
        n.rel_list = sorted(n.rel_list)
        # 被merge的點就從nodes中消失了，本來的位置設為None
        for ch in n.mergelist:
            n.children.remove(ch)
            n.children.extend(nodes[ch].children)
            for i in nodes[ch].children:
                nodes[i].parent = n
            nodes[ch] = None
        n.children = sorted(n.children)
        # 如果任一子節點無法被merge，則就不該再往上merge，應該設自己merged=False
        # 但如果parent只有自己一個child，則仍然會merge
        if len(n.parent.children) > 1 and any([not nodes[ch].merged for ch in n.children]):
            n.merged = False
            if n.id in n.parent.mergelist:
                n.parent.mergelist.remove(n.id)

    def execute_merge(self):
        if not self.isMerging:
            return
        self.recursive_merge(self.nodes, self.nodes[0].children[0])
        nonempty = [i for i, n in enumerate(self.nodes) if n] + [0]
        for i, nid in enumerate(nonempty[1:]):
            self.nodes[nid].next = self.nodes[nonempty[i + 1]]
            self.nodes[nid].prev = self.nodes[nonempty[i - 1]]

    def find_possible_root(self, tree, collect):
        if tree['rel'] in ('ROOT', 'dislocated'):
            collect.append(tree)
        for child in tree['children']:
            self.find_possible_root(child, collect)

    def node_chunking(self):
        chunk, chunk_buf, rel_list, pos_list = [], [], [], []
        nodes = [n for n in self.nodes[1:] if n is not None]
        if len(nodes) > 1 and (nodes[1].pos in chunking_postag['VP'] or nodes[1].pos in chunking_postag['NP']):
            chunk_buf.append('')

        for n in nodes:
            nodeIsVP = any([pos in chunking_postag['VP'] for _, pos in n.pos_list])
            nodeIsNP = any([pos in chunking_postag['NP'] for _, pos in n.pos_list])
            if nodeIsVP or nodeIsNP:  # 如果是名詞片語或動詞片語，則加入chunk_buf
                chunk_buf.append(n.name)
                pos_list.extend(n.pos_list)
                rel_list.extend(n.rel_list)
            elif chunk_buf:  # 非名詞或非動詞
                if n.rel == 'punct' and n.next.id > 0:
                    # parataxis 通常語意是獨立的
                    if n.next.rel == 'parataxis':
                        append_chunk = False
                    # 如果chunk本身不完整(缺主詞或動詞)，則應把逗號視為chunk的一部分
                    elif any([rel.count('subj') > 0 for _, rel in n.next.rel_list]):
                        append_chunk = True
                    else:  # 一般情況下，逗號會隔開chunk
                        append_chunk = False
                else:
                    append_chunk = False  # 非名詞或非動詞預設會隔開chunk
                if append_chunk:
                    chunk_buf.append(n.name)
                    pos_list.extend(n.pos_list)
                    rel_list.extend(n.rel_list)
                else:  # chunk被中斷後，就將單字合併為字串
                    chunk.append(' '.join(chunk_buf))  # collect previous chunk
                    chunk_buf, rel_list, pos_list = [], [], []
        if chunk_buf:
            chunk.append(' '.join(chunk_buf))
        return self.clear_word_space(chunk)

    def recursive_tree_chunking(self, tree, depth, depth_node):
        if depth >= len(depth_node):
            return
        depth_node[depth].append(tree)
        for child in tree['children']:
            self.recursive_tree_chunking(child, depth + 1, depth_node)

    def tree_chunking(self, tree, max_depth=6):
        depth_node = [list() for _ in range(max_depth)]
        self.recursive_tree_chunking(tree, 0, depth_node)
        nodes = []
        chunk = []
        for depth in range(max_depth):
            nodes.extend(depth_node[depth])
            names = []
            for n in nodes:
                # 任何comp結尾的關係都視為必須，如果child中有以comp關係連接
                # 但child還沒有加入nodes，則放棄這個點
                if any([child['rel'] in must_connect_rel and child not in nodes
                       for child in n['children']]):
                   continue
                # 如果遇到連接詞，先產生一個沒有連接的chunk
                # if n['rel'] == 'cc':
                #     chunk.append(' '.join([name for _, name, _ in sorted(names)]))
                names.append((n['id'], n['name'], n['rel']))
            names = sorted(names)
            # 如果有連接詞'cc'，就一定要有連接的部分'conj'
            for i in range(len(names) - 1):
                if names[i][2] == 'cc' and names[i + 1][2] != 'conj':
                    names[i] = (0, '', '')
            while names and names[0][2] in ('punct', 'mark'):
                del names[0]
            chunk.append(' '.join([name for _, name, _ in names]))
        return chunk

    def chunking(self):
        # return list(set(self.node_chunking() + self.tree_chunking(self.tree)))
        roots = []
        self.find_possible_root(self.tree, roots)
        chunks = []
        for root in roots:
            chunks.extend(self.tree_chunking(root))
        chunks = [ch for ch in set(chunks) if ch]
        return chunks
