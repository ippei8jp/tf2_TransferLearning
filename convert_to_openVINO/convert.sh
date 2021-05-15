#!/usr/bin/bash

TARGET=COCO
# TARGET=HAND
# TARGET=VOC

if [ "${TARGET}" == "COCO" ]; then

  # COCO
  if [ ! -e ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 ]; then
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz -O - | tar xzvf -
  fi

  MODEL_NAME=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
  SAVED_MODEL=./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model
  FROZEN_MODEL=./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/simple_frozen_graph.pb
  CONFIG_FILE=./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config
  LABELS_FILE=../ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/label_map.txt
  TRANS_CONFIG=/opt/intel/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v2.0.json

elif [ "${TARGET}" == "HAND" ]; then

  # hand_detect
  # このモデルの変換はエラーになる。今後の修正に期待。
  MODEL_NAME=ssd_mobilenet_v2_hand_detect
  SAVED_MODEL=../hand_detect/saved_model
  FROZEN_MODEL=../hand_detect/simple_frozen_graph.pb
  CONFIG_FILE=../hand_detect/pipeline.config
  LABELS_FILE=../hand_detect/label_map.txt
  TRANS_CONFIG=/opt/intel/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v2.0.json

else

  # voc_detect
  # このモデルの変換はエラーになる。今後の修正に期待。
  MODEL_NAME=ssd_mobilenet_v2_voc_detect
  SAVED_MODEL=../voc_detect/saved_model
  FROZEN_MODEL=../voc_detect/simple_frozen_graph.pb
  CONFIG_FILE=../voc_detect/pipeline.config
  LABELS_FILE=../voc_detect/label_map.txt
  TRANS_CONFIG=/opt/intel/openvino_2021/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v2.0.json

fi

# frozen modelへの変換
# (saved model からの変換はエラーになるのでfrozen modelに変換してから実行する)
python freeze.py ${SAVED_MODEL} ${FROZEN_MODEL} 

if [ $? -ne 0 ]; then
  # エラー
  exit 1
fi

# モデルオプティマイザオプション設定
OPTIONS=" --framework=tf"
# OPTIONS+=" --log_level=DEBUG"       # logレベル
OPTIONS+=" --data_type=FP16"
OPTIONS+=" --transformations_config=${TRANS_CONFIG}"
OPTIONS+=" --reverse_input_channels"
OPTIONS+=" --input_shape=[1,320,320,3]"
OPTIONS+=" --input=input_tensor"
OPTIONS+=" --tensorflow_object_detection_api_pipeline_config=${CONFIG_FILE}"
OPTIONS+=" --model_name=${MODEL_NAME}"
# OPTIONS+=" --saved_model_dir=${SAVED_MODEL}"  # saved_modelからの変換はエラーになる
OPTIONS+=" --input_model=${FROZEN_MODEL}"       # saved_model指定時はこのオプションをコメントアウト 
OPTIONS+=" --output_dir=./_IR/${MODEL_NAME}/FP16"

# 実行
# echo ${OPTIONS}
/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py ${OPTIONS}

# ラベルファイルもコピー
if [ -f ${LABELS_FILE} ]; then
    cp ${LABELS_FILE} ./_IR/${MODEL_NAME}/FP16/${MODEL_NAME}.labels
fi

# object_detectionモジュールがインストールされていれば以下でも変換可能
# python convert_labels.py ${LABELS_FILE} ./_IR/${MODEL_NAME}/FP16/${MODEL_NAME}.labels
