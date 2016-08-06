# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("input_img")
parser.add_argument("output_img")
args = parser.parse_args()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import model

extractor = model.FaceExtractor()
try:
    face_img = cv2.cvtColor(
        extractor.extract(args.input_img) * 256,
        cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(args.output_img, face_img)
except Exception as e:
    print(e)
