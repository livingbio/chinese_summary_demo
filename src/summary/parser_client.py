import requests
import json


class Parser(object):
    def __init__(self, lang='en', server='localhost', port=8010):
        self.url = 'http://{}:{}'.format(server, port)
        self.lang = lang

    def parse(self, list_of_sent):
        if isinstance(list_of_sent, basestring):
            list_of_sent = [list_of_sent]
        data = {'sentences': u'##'.join(list_of_sent)}
        url = '{}/parser/{}'.format(self.url, self.lang)
        r = json.loads(requests.post(url, data=data).text)
        return r['nodes']

    def tag(self, list_of_sent):
        if isinstance(list_of_sent, basestring):
            list_of_sent = [list_of_sent]
        data = {'sentences': u'##'.join(list_of_sent)}
        url = '{}/tagger/{}'.format(self.url, self.lang)
        r = json.loads(requests.post(url, data=data).text)
        return r['nodes']
