# 概要
Tensorflow2環境でのSSDモデルの転移学習の手順を試してみた。  
検索でヒットするのはTensorflow1環境のものがほとんどなので、やっと見つかったTensorflow2の手順を試したときのメモ。  

# 参考
以下のサイトをトレースしています。  
< https://towardsdatascience.com/train-an-object-detector-using-tensorflow-2-object-detection-api-in-2021-a4fed450d1b9>


# 転移学習の実行
HandDetector.ipynb をGoogle Colaboratory で実行

# 実行完了をひたすら待つ
試したときは3時間くらｋかかった
コマンド実行終了した後、ほったらかしておくと
「あんたロボットと違う？」と聞かれ、ちゃんと答えないと接続切られちゃうので注意。  
(スタートして寝たり出掛けたりしちゃうと、せっかく学習終了したのに消えちゃうことも)  

最初のセルを実行してカレントをGoogleDriveにしておくと接続切られても結果は残ってる。  
でも、空き容量ないと動かないので注意。  

## 生成済みモデルファイルのダウンロード
hand_detect_XXXXXXXX_XXXXXX.zip をダウンロード

# ローカル環境でのテスト

## このリポジトリをcloneする

## python仮想環境設定
```
pyenv virtualenv 3.7.10 hand_detect
pyenv local hand_detect
pip install --upgrade pip setuptools
```

## pythonモジュールのインストール
```
git clone --depth 1 https://github.com/tensorflow/models
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .

cd ../../
```

## ダウンロードしたモデルファイルを展開する
hand_detect_XXXXXXXX_XXXXXX.zip を展開  

## テスト

テストディレクトリに移動  
```
cd _test

sh _test.sh
```
または、任意の画像ファイルを指定して以下を実行
```
python hand_detect.py <<JPEGファイル>>
```
