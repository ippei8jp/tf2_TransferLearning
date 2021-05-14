import os
import sys
import cv2

import numpy as np
import tensorflow as tf

from PIL import Image
# from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# ラベルマップのロード
PATH_TO_LABELS = '../inference/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# モデルのロード
detection_model = tf.saved_model.load("../inference/saved_model")

# Check the model's input signature, it expects a batch of 3-color images of type uint8:
print('-------------------------')
print(detection_model.signatures['serving_default'].inputs)
print('-------------------------')

# And returns several outputs:
print('=========================')
print(detection_model.signatures['serving_default'])
print('=========================')
print(detection_model.signatures['serving_default'].output_dtypes)
print(detection_model.signatures['serving_default'].output_shapes)
print('=========================')
print(detection_model.signatures.keys())

# 認識処理関数
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

# 認識処理と表示
def show_inference(model, image_path):
  # 画像の読み込み
  image_np = np.array(Image.open(image_path))
  
  # 認識実行
  output_dict = run_inference_for_single_image(model, image_np)
  
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
  
  # 表示
  # display(Image.fromarray(image_np))
  new_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
  cv2.imshow("Detection Results", new_image)  
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# 実行
show_inference(detection_model, sys.argv[1])

