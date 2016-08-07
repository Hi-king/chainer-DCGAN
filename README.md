DCGAN vectorizer
============================

DCGANの実装や解説は <https://github.com/mattya/chainer-DCGAN> を参照してください。
ここでは、画像空間からパラメタ空間に変換するVectorizerを使って遊ぶサンプルを公開します。

## Vectorizer

学習済みvectorizerモデルをダウンロード

```
wget https://www.dropbox.com/s/ri63mw1yzhupx3v/vectorizer_model_9990000.h5?dl=0
```

あるいは、自分で学習する

```
python bin/train_vectorizer.py generator_model.h5 100000000 --gpu=-1 --out_dir=output
```

## Morphing

2つの画像の中間画像を生成する。

<img src="https://raw.githubusercontent.com/Hi-king/chainer-DCGAN/master/result/sakurako_himawari/cat.png">

```
python bin/morphing.py original/generator_model.h5 vectorizer_model_9990000.h5 img/sakurako.jpg img/himawari.jpg --step=20 --out_dir=result/sakurako_himawari
montage result/sakurako_himawari/morphing*.png -tile x1 result/sakurako_himawari.png
```

## Average

ディレクトリ内の画像の中間画像を生成する。

<img src="https://raw.githubusercontent.com/Hi-king/chainer-DCGAN/master/result/lovelive/average.png">

```
python bin/search_tag.py ラブライブ img/lovelive
python bin/average.py original/generator_model.h5 vectorizer_model_9990000.h5 img/lovelive --out_dir=./result/lovelive
```
