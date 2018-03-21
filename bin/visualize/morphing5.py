import argparse

import chainer
import os
import sys
import cv2
import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import model

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import morphing


def main(args):
    resize = (96, 96)
    generator = model.Generator()
    chainer.serializers.load_hdf5(args.generator_model_file, generator)
    vectorizer = model.Vectorizer()
    chainer.serializers.load_hdf5(args.vectorizer_model_file, vectorizer)

    vectors = morphing.vectorize(args.target_imgs, vectorizer, resize, args.margin)


    SIZE = 7
    result_img = numpy.zeros((resize[0] * SIZE, resize[1] * SIZE, 3), dtype=numpy.uint8)

    c = vectors[4]
    print(args.target_imgs)
    for ix, iy, reverse_x, reverse_y in [
        (1, 0, False, False),
        (1, 2, False, True),
        (3, 2, True, False),
        (3, 0, True, True),
    ]:
        a, b = vectors[ix], vectors[iy]
        ylen = SIZE // 2 + 1
        for y in range(ylen):
            xlen = SIZE - y * 2
            for x in range(xlen):
                rx = (float(x) / (xlen - 1)) if x > 0 else 0
                ry = float(y) / (ylen - 1)
                generated = generator(
                    a * (1 - rx) * (1 - ry)
                    + b * (rx) * (1 - ry)
                    + c * ry,
                    test=True
                )
                generated_img = morphing.to_img(generated.data)
                print(generated_img.dtype)

                coordinate_x = resize[0] * (x + y)
                coordinate_y = resize[1] * y
                if reverse_x:
                    coordinate_x = result_img.shape[0] - coordinate_x - resize[0]
                    coordinate_y = result_img.shape[1] - coordinate_y - resize[1]

                if reverse_y:
                    coordinate_x, coordinate_y = coordinate_y, coordinate_x

                result_img[coordinate_x:coordinate_x + resize[0], coordinate_y:coordinate_y + resize[1]] = generated_img
    cv2.imwrite("morphing5.png", result_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("generator_model_file")
    parser.add_argument("vectorizer_model_file")
    parser.add_argument("target_imgs", nargs=5)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--margin", type=float, default=0.3, help="rate of facial region")
    args = parser.parse_args()
    main(args)
