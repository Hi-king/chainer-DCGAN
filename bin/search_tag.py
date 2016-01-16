# -*- coding: utf-8 -*-
import sys
import os
import urllib2
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import nicosearch

parser = argparse.ArgumentParser()
parser.add_argument("target_tag")
parser.add_argument("image_dir")
args = parser.parse_args()

if not os.path.exists(args.image_dir):
    os.mkdir(args.image_dir)

search = nicosearch.NicoSearch()

for line in search.find_by_tag(args.target_tag):
    cmsid = line["cmsid"][2:]
    url = "http://lohas.nicoseiga.jp//thumb/{}i".format(cmsid)
    f = open(os.path.join(args.image_dir, cmsid+".jpg"), "w+")
    f.write(urllib2.urlopen(url).read())
