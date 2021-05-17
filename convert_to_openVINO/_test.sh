TARGET=COCO
# TARGET=HAND
# TARGET=VOC

# pyenvでpythonのバージョンを切り替えたときの問題の対策  ==================================
## 現在のpythonのバージョン取得
# TMP_PYVER=`python -V |  sed -e "s/^.*\(3.[0-9]\{1,\}\).*$/\1/g"`
TMP_PYVER=`python -c "import sys; v = sys.version_info; print(f'{v[0]}.{v[1]}')"`
## PYTHONPATHの該当箇所を置換
export PYTHONPATH=`echo $PYTHONPATH  | sed -e "s/\/python3\.[0-9]\{1,\}/\/python${TMP_PYVER}/g"`
# =========================================================================================

# サンプル画像ダウンロード
if [ "${TARGET}" == "COCO" ]; then

  MODEL_NAME=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
  JPEG_FILE=blueangels.jpg
  if [ ! -e ${JPEG_FILE} ]; then
    wget https://farm5.staticflickr.com/4151/5097393934_7e0a2873bf_z.jpg -O ${JPEG_FILE}
  fi

elif [ "${TARGET}" == "HAND" ]; then

  MODEL_NAME=ssd_mobilenet_v2_hand_detect
  JPEG_FILE=hand1.jpg
  if [ ! -e ${JPEG_FILE} ]; then
    wget https://cdn.amebaowndme.com/madrid-prd/madrid-web/images/sites/483796/1357355de6edbc4c4b54d22faf0b0756_ce052e9b134a9dbb047a8e17c890832a.jpg -O ${JPEG_FILE}
  fi

else

  MODEL_NAME=ssd_mobilenet_v2_hand_detect
  JPEG_FILE=airplane.jpg
  if [ ! -e ${JPEG_FILE} ]; then
    wget https://prtimes.jp/i/6067/298/resize/d6067-298-418042-0.jpg -O ${JPEG_FILE}
  fi

fi
MODEL_FILE=./_IR/${MODEL_NAME}/FP16/${MODEL_NAME}.xml

# テスト実行
python ov_object_detection_ssd.py -m ${MODEL_FILE}  -i ${JPEG_FILE} 
