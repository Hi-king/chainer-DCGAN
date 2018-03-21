import argparse
import os

import chainer
import chainer.optimizers
import chainer.serializers
import cv2
import math
import numpy
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import model


def vectorize(target_imgs, vectorizer, resize, margin, predetected=False):
    extractor = model.FaceExtractor(margin=margin, resize=resize)
    ## vectorize
    vectors = []
    for target_img in target_imgs:
        print(target_img)
        if predetected:
            face_img = cv2.imread(target_img, cv2.IMREAD_UNCHANGED)
            if face_img.shape[2] == 4:
                face_img[face_img[:, :, 3] == 0] = (255, 255, 255, 255)
            face_img = cv2.resize(face_img, resize, interpolation=cv2.INTER_AREA)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = face_img.astype(numpy.float32) / 256
        else:
            face_img = extractor.extract(target_img)
        face_img_var = chainer.Variable(
            numpy.array([face_img.transpose(2, 0, 1) * 2 - 1.0]))
        vector = vectorizer(face_img_var)
        vectors.append(vector)
    return vectors


def clip_img(x):
    return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def to_img(x):
    img = ((numpy.vectorize(clip_img)(x[0, :, :, :]) + 1) / 2).transpose(1, 2, 0)
    return cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR)


def save(x, filepath):
    cv2.imwrite(
        filepath,
        to_img(x)
    )


def main(args):
    resize = (96, 96)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    generator = model.Generator()
    chainer.serializers.load_hdf5(args.generator_model_file, generator)
    vectorizer = model.Vectorizer()
    chainer.serializers.load_hdf5(args.vectorizer_model_file, vectorizer)

    vectors = vectorize(args.target_imgs, vectorizer, resize, args.margin)

    for ss in range(args.step * (len(args.target_imgs) - 1) * 2):
        if ss >= args.step * (len(args.target_imgs) - 1):
            s = args.step * (len(args.target_imgs) - 1) * 2 - ss
        else:
            s = ss

        i = int(math.floor(s / args.step))
        res = float(s % args.step) / (args.step - 1)
        j = i + 1 if not s == args.step * (len(args.target_imgs) - 1) else i
        print(i, j, res)
        generated = generator(
            vectors[i] * (1 - res)
            + vectors[j] * (res),
            test=True
        )
        save(generated.data, os.path.join(args.out_dir, "morphing{:04d}.png".format(ss)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("generator_model_file")
    parser.add_argument("vectorizer_model_file")
    parser.add_argument("target_imgs", nargs="+")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--predetected", action="store_true")
    parser.add_argument("--margin", type=float, default=0.3, help="rate of facial region")
    args = parser.parse_args()
    main(args)
