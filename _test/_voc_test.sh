# サンプル画像ダウンロード
if [ ! -e voc_a.jpg ]; then
  wget https://prtimes.jp/i/6067/298/resize/d6067-298-418042-0.jpg -O voc_a.jpg
  wget https://www.kic-car.ac.jp/theme/kic_school/img/taisho/ph-society001.jpg -O voc_b.jpg
fi

python _test.py ../voc_detect/label_map.pbtxt ../voc_detect/saved_model ./voc_a.jpg
# python _test.py ../voc_detect/label_map.pbtxt ../voc_detect/saved_model ./voc_b.jpg
