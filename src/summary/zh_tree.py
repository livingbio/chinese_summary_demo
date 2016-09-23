# -*- coding: utf-8 -*-
from syntaxnet_wrapper import parser
import re
from libvec import zh

rel_should_merge = {
    'acomp': 'adjectival complement',  # she looks (beautiful)
    'advcl': 'adverbial clause modifier',
    'advmod': 'adverb modifier',  # (Genetically) modified food”
    # 'agent': 'agent',  # He killed by (the police)
    'amod': 'adjectival modifier',
    # 'appos': 'appositional modifier',  # Sam, (my brother), arrived
    'aux': 'auxiliary',  # He (should) leave
    'auxpass': 'passive auxiliary',  # He (was) fired
    # 'cc': 'coordination',  # Bill is big (and) honest
    # 'ccomp': 'clausal complement',  # He says (that you like to swim)
    # 'conj': 'conjunct',  # Bill is big and (honest)
    'cop': 'copula',  # Bill (is) big
    'csubj': 'clausal subject',  # (What she said) is not true
    'csubjpass': 'clausal passive subjec',  # (That she lied) was suspected
    'det': 'determiner',  # (The) man is here
    'dep': 'dependant',  # unknown dependant
    # 'discourse': 'discourse element',  # (oh) ! you are here!
    'dobj': 'direct object',  # She gave me (a raise)
    'expl': 'expletive',  # (There) is a dog
    'goeswith': 'goes with',  # They come here (with out) legal permission
    'iobj': 'indirect object',  # She gave (me) a raise
    # 'mark': 'marker',  # Forces engaged in fighting (after insurgents attacked)
    'mwe': 'multi-word expression',  # I like dogs (as well as) cats
    'neg': 'negation modifier',  # Bill is (not) a scientist
    'nn': 'noun compound modifier',  # (Oil price) is high
    'npadvmod': 'noun phrase as adverbial modifier',  # 6 (feet) long
    'nsubj': 'nominal subject',  # The (baby) is cute
    'nsubjpass': 'passive nominal subject',  # (Dole) was defeated by Clinton
    'num': 'numeric modifier',  # Sam ate (3) sheep
    'number': 'element of compound number',  # I lost $ (3.2) billion
    # 'parataxis': 'parataxis',  # The guy, (John said), left early in the morning
    'pcomp': 'prepositional complement',  # They heard about (you missing classes)
    'pobj': 'object of a preposition',  # I sat on (the chair)
    'poss': 'possession modifier',  # (their) offices
    'possessive': 'possessive modifier',  # Bill (’s) clothes
    # 'preconj': 'preconjunct',  # (Both) the boys and the girls are here
    'predet': 'predeterminer',  # (All) the boys are here
    'prep': 'prepositional modifier',  # I saw a cat (in a hat)
    'prepc': 'prepositional clausal modifier',  # He purchased it (without paying a premium)
    'prt': 'phrasal verb particle',  # They shut (down) the station
    # 'punct': 'punctuation',  # Go home (!)
    'quantmod': 'quantifier phrase modifier',  # (About) 200 people came to the party
    'rcmod': 'relative clause modifier',  # I saw the book which (you bought)
    # 'ref' : 'referent',  # I saw the book (which) you bought
    # 'tmod': 'temporal modifier',  # I swam in the pool last (night)
    'vmod': 'reduced non-finite verbal modifier',  # I don’t have anything (to say to you)
    'xcomp': 'open clausal complement',  # I am ready (to leave)
    'xsubj': 'controlling subject',  # (Tom) likes to eat fish
}

words_with_apos = [
    'arent', 'cant', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt',
    'havent', 'isnt', 'mustnt', 'shouldnt', 'wasnt', 'werent', 'wont', 'wouldnt',
]

words_with_space = [
    " 're ", " n't ", " 've ", " 's ", " 'll ", " 'd ",
]

chunking_postag = {
    'VP': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    'NP': {'NN', 'NNS', 'NNP', 'NNPS', 'WDT'},
}

must_connect_rel = {
    'acomp', 'ccomp', 'pcomp', 'xcomp', 'dobj', 'iobj', 'pobj',
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


class DependencyTree(object):
    def __init__(self, sentence, **kwargs):
        if 'lang' in kwargs:
            self.parser = parser[kwargs['lang']]
        else:
            self.parser = parser['zh']
        if 'merging' in kwargs:
            self.isMerging = kwargs['merging']
        else:
            self.isMerging = True
        if 'name_with_pos' in kwargs:
            self.isNameWithPOS = kwargs['name_with_pos']
        else:
            self.isNameWithPOS = False

        raw = self.parser.query(' '.join(zh.tw_segment(sentence)))
        n_nodes = len(raw) + 1
        nodes = [TreeNode() for _ in range(n_nodes)]  # nodes[0] is dummy root
        for n in raw:
            i = int(n[0])
            p = int(n[6])
            nodes[i].id = int(n[0])
            nodes[i].name = n[1]
            if nodes[i].name in words_with_apos:
                nodes[i].name = nodes[i].name.replace('nt', 'n\'t')
            nodes[i].pos = n[4]
            nodes[i].parent = nodes[p]
            nodes[i].rel = n[7]
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
            # 如果是"and xxx"的句型，也要merge
            elif n.rel == 'cc' and n.next.rel == 'conj' and not n.children and not n.next.children:
                n.parent.mergelist.extend([n.id, n.id + 1])
            # 一般","不會merge，但遇到"$ 1 , 000"時則要
            if n.rel == 'number' and n.prev.pos == ',':
                n.parent.mergelist.append(n.id - 1)

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
        if not n.mergelist:
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
        if tree['rel'] in ('parataxis', 'ROOT'):
            collect.append(tree)
        for child in tree['children']:
            self.find_possible_root(child, collect)

    def clear_word_space(self, chunk):
        ret_chunk = []
        for i, ch in enumerate(chunk):
            if len(ch) == 0:
                continue
            for w in words_with_space:  # 讓顯示較美觀，合併 "ca n't" 為"can't"
                ch = ch.replace(w, w[1:])
            ch = re.sub('[,:;\.\!\?] ([,:;\.\!\?])', '\\1', ch)
            ch = re.sub(' [,:;]$', '', ch)
            ret_chunk.append(ch)
        return ret_chunk

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
            rel_list = []
            for n in nodes:
                # 任何comp結尾的關係都視為必須，如果child中有以comp關係連接
                # 但child還沒有加入nodes，則放棄這個點
                if any([child['rel'] in must_connect_rel and child not in nodes
                        for child in n['children']]):
                    continue
                # 如果遇到連接詞，先產生一個沒有連接的chunk
                if n['rel'] == 'cc':
                    chunk.append(' '.join([name for _, name in sorted(names)]))
                names.append((n['id'], n['name']))
                rel_list.append(n['rel'])
            # 如果有連接詞'cc'，就一定要有連接的部分'conj'
            for i, rel in enumerate(rel_list[:-1]):
                if rel == 'cc' and rel_list[i + 1] != 'conj':
                    names[i] = (0, '')
            chunk.append(' '.join([name for _, name in sorted(names)]))
        return self.clear_word_space(chunk)

    def chunking(self):
        # return list(set(self.node_chunking() + self.tree_chunking(self.tree)))
        roots = []
        self.find_possible_root(self.tree, roots)
        chunks = []
        for root in roots:
            chunks.extend(self.tree_chunking(root))
        chunks = list(set(chunks))
        return chunks
