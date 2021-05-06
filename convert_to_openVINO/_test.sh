TARGET=COCO
# サンプル画像ダウンロード
if [ "${TARGET}" == "COCO" ]; then

  MODEL_NAME=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
  JPEG_FILE=blueangels.jpg
  if [ ! -e ${JPEG_FILE} ]; then
    wget https://farm5.staticflickr.com/4151/5097393934_7e0a2873bf_z.jpg -O ${JPEG_FILE}
  fi

else

  MODEL_NAME=ssd_mobilenet_v2_hand_detect
  JPEG_FILE=blueangels.jpg
  if [ ! -e ${JPEG_FILE} ]; then
    wget https://cdn.amebaowndme.com/madrid-prd/madrid-web/images/sites/483796/1357355de6edbc4c4b54d22faf0b0756_ce052e9b134a9dbb047a8e17c890832a.jpg -O ${JPEG_FILE}
    # wget https://cdn.amebaowndme.com/madrid-prd/madrid-web/images/sites/483796/564b6ca69e9022aa1977f335a148a05a_2d642c807aaf8f5b972a0a406903447d.jpg -O b.jpg
  fi

fi
MODEL_FILE=./_IR/${MODEL_NAME}/FP16/${MODEL_NAME}.xml

# テスト実行
python ov_object_detection_ssd.py -m ${MODEL_FILE}  -i ${JPEG_FILE} 
