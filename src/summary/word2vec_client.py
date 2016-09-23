import requests
import json
import numpy as np


class Word2Vec(object):
    def __init__(self, lang='en', server='localhost', port=8080):
        self.url = 'http://{}:{}'.format(server, port)
        self.lang = lang

    def wordvec(self, list_of_word):
        if isinstance(list_of_word, basestring):
            list_of_word = [list_of_word]
        data = {'sentences': u'##'.join(list_of_word)}
        url = '{}/words_vec/{}'.format(self.url, self.lang)
        r = json.loads(requests.post(url, data=data).text)
        return np.array(r['vector'])

    def sentvec(self, list_of_sent):
        if isinstance(list_of_sent, basestring):
            list_of_sent = [list_of_sent]
        data = {'sentences': u'##'.join(list_of_sent)}
        url = '{}/sents_vec/{}'.format(self.url, self.lang)
        r = json.loads(requests.post(url, data=data).text)
        return np.array(r['vector'])

    def oovword(self, list_of_sent):
        if isinstance(list_of_sent, basestring):
            list_of_sent = [list_of_sent]
        data = {'sentences': u'##'.join(list_of_sent)}
        url = '{}/sents_oov/{}'.format(self.url, self.lang)
        r = json.loads(requests.post(url, data=data).text)
        return r['oov']
