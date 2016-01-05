Morphing
============================

DCGANの実装や解説は <https://github.com/mattya/chainer-DCGAN> を参照してください

vectorizerモデルをダウンロード

```
wget https://www.dropbox.com/s/natlidgg6ppnpnv/vectorizer_model_4790000.h5?dl=0
```

morphing

```
python bin/morphing.py generator_model.h5 vectorizer_model_4790000.h5 img/sakurako.jpg img/himawari.jpg --step=20 --out_dir=result/sakurako_himawari
```

