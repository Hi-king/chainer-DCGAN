import os
import argparse
import chainer
import chainer.serializers
import chainer.optimizers
import numpy
import pylab
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import model

parser = argparse.ArgumentParser()
parser.add_argument("generator_model_file")
parser.add_argument("vectorizer_model_file")
parser.add_argument("target_img_0")
parser.add_argument("target_img_1")
parser.add_argument("--out_dir", default=".")
parser.add_argument("--step", type=int, default=5)
parser.add_argument("--margin", type=float, default=0.3, help="rate of facial region")
args = parser.parse_args()
xp = numpy

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

generator = model.Generator()
chainer.serializers.load_hdf5(args.generator_model_file, generator)
extractor = model.FaceExtractor(margin=args.margin)
vectorizer = model.Vectorizer()
chainer.serializers.load_hdf5(args.vectorizer_model_file, vectorizer)


## vectorize
vectors = []
for target_img in [args.target_img_0, args.target_img_1]:
    face_img = extractor.extract(target_img)
    face_img_var = chainer.Variable(
        numpy.array([face_img.transpose(2,0,1)*2 - 1.0]))
    #pylab.imshow(face_img)
    #pylab.show()
    vector = vectorizer(face_img_var)
    vectors.append(vector)

def clip_img(x):
    return numpy.float32(-1 if x<-1 else (1 if x>1 else x))

def save(x, filepath):
    img = ((numpy.vectorize(clip_img)(x[0,:,:,:])+1)/2).transpose(1,2,0)
    pylab.imshow(img)
    pylab.axis('off')
    # pylab.savefig(filepath)
    pylab.show()

    print(img.shape)
    cv2.imwrite(
        filepath,
        cv2.cvtColor(img*256, cv2.COLOR_RGB2BGR)
    )

diff = vectors[1] - vectors[0]
for s in xrange(args.step):
    generated = generator(
        vectors[0]*float(s)/(args.step-1)
        + vectors[1]*float(args.step-s-1)/(args.step-1),
        test=True
    )
    save(generated.data, os.path.join(args.out_dir, "morphing{}.png".format(s)))
