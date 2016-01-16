# -*- coding: utf-8 -*-
import urllib2
import json

class NicoSearch(object):
    def find_by_tag(self, tag_string, size=100):
        endpoint = "http://api.search.nicovideo.jp/api/"
        for index_from in xrange(0, size, 100):
            query_size = min(100, size-index_from)
            values = {
                "query": tag_string,
                "service":["illust"],
                "search":["tags_exact"],
                "join": ["cmsid","title","view_counter"],
                "from":index_from,
                "size":query_size,
                "sort_by":"view_counter",
                "issuer":"apiguide",
                "reason":"ma10",
            }
            request = urllib2.Request(endpoint)
            request.add_header('Content-Type', 'application/json')
            response = urllib2.urlopen(request, json.dumps(values))
            for line in response.read().split("\n"):
                if len(line) == 0: continue
                line = json.loads(line)
                if not "type" in line.keys():
                    print line
                if line["type"] == "hits":
                    if not "values" in line: continue
                    for value in line["values"]:
                        yield value
