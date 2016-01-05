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
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()
xp = numpy

generator = model.Generator()
chainer.serializers.load_hdf5(args.generator_model_file, generator)

extractor = model.FaceExtractor()

vectorizer = model.Vectorizer()
chainer.serializers.load_hdf5(args.vectorizer_model_file, vectorizer)

z_data = (xp.random.uniform(-1, 1, (1, 100)).astype(numpy.float32))
z = chainer.Variable(z_data)
x = generator(z, test=True)

reconstructed = vectorizer(x)
regenerated = generator(reconstructed, test=True)

def clip_img(x):
    return numpy.float32(-1 if x<-1 else (1 if x>1 else x))

def save(x, filepath):
    # pylab.rcParams['figure.figsize'] = (22.0,22.0)
    # pylab.rcParams['figure.figsize'] = (1,1)
    img = ((numpy.vectorize(clip_img)(x[0,:,:,:])+1)/2).transpose(1,2,0)
    # pylab.imshow(img)
    # pylab.axis('off')
    # pylab.savefig(filepath,
    #               pad_inches=0)

    cv2.imwrite(
        filepath,
        cv2.cvtColor(img*256, cv2.COLOR_RGB2BGR)
    )

save(x.data, os.path.join(args.out_dir, "constructed.png"))
save(regenerated.data, os.path.join(args.out_dir, "reconstructed.png"))
