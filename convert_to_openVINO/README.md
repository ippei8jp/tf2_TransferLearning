# 概要
Tensorflow2環境でのSSDモデルをopenVINOのモデルに変換する。  
転移学習の結果を変換したかったが、エラーになってうまく動かない。  
変換手順が正しいことを確認するために、転移学習のベースにしたモデルで変換操作を試してみた。  
実行には、***openVINO 2021.3以降***が必要。  


# 参考
Saved modelをFrozen modelに変換する：
<https://github.com/openvinotoolkit/openvino/issues/4830>

# モデルの変換
```
bash convert.sh
```
``_IR``ディレクトリ以下に変換モデルができる。  

# テスト
テスト用画像ファイルをダウンロードして、認識処理を行う。    

```
bash _test.sh
```

