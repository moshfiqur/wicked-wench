'''
Sample run command:
python predict_n_crop.py \
    --image '../images/mobile_1.jpg' \
    --tf-model 'model' \
    --tf-research '/Users/sparrow/Learning/machine-learning/tensorflow-models-zoo/research' \
    --label-path '../data'
'''

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
import argparse
import cv2

# This was required to fix the 'python was not installed as framework' error
# with matplotlib on Mac OSX
import matplotlib as mpl
mpl.use('TkAgg')

parser = argparse.ArgumentParser()
# Test image path to run the prediction on
parser.add_argument('--image', dest='image_path', required=True)

# Path to the imagenet model .tar.gz file
parser.add_argument('--tf-model', dest='tf_model_path', required=True)

# Path to the tensorflow research models and utils: https://github.com/tensorflow/models
# Not needed if the path is exist in system PYTHONPATH
parser.add_argument('--tf-research', dest='tf_research_path', required=True)

# Path to object labels for coco dataset
parser.add_argument('--label-path', dest='label_path', required=True)

args = parser.parse_args()

IMAGE_PATH = str(StringIO(args.image_path).getvalue())
TF_MODEL_PATH = str(StringIO(args.tf_model_path).getvalue())
TF_RESEARCH_PATH = str(StringIO(args.tf_research_path).getvalue())
LABEL_PATH = str(StringIO(args.label_path).getvalue())

# ROOT_PATH = '/Users/moshfiqur/Learning/machine-learning/object-detection-playground'
# DATA_PATH = os.path.join(ROOT_PATH, 'data')
# TF_MODELS_PATH = '/Users/moshfiqur/Learning/machine-learning/tf-models/models'
# TEST_IMAGES_PATH = '/Users/moshfiqur/Learning/machine-learning/object-detection-playground/images'

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append(os.path.join(TF_MODELS_PATH, 'research'))
sys.path.append(TF_RESEARCH_PATH)
sys.path.append(os.path.join(TF_RESEARCH_PATH, 'object_detection'))
# sys.path.append(ROOT_PATH)

# print(sys.path)

# print(tf.__version__)

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')

from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = os.path.join(TF_MODEL_PATH, MODEL_NAME + '.tar.gz')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_CKPT_PATH = os.path.join(MODEL_NAME + '/frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
LABEL_FILE = os.path.join(LABEL_PATH, 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Extract the model file to get the frozen detection graph in it
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# Initialize the graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL_CKPT_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Initialize the label map
label_map = label_map_util.load_labelmap(LABEL_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Images can be sent as | separated string
test_images = IMAGE_PATH.split('|')

# Perform the actual detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        for image_path in test_images:
            if '.jpg' not in image_path:
                continue
            
            # Where the cropped images will be saved
            save_path = os.path.join('images', 'cropped', os.path.basename(image_path))
            
            image = Image.open(image_path)
            
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = cv2.imread(image_path)

            height = image_np.shape[0]
            width = image_np.shape[1]

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            # Actual detection.
            (result_boxes, result_scores, result_classes, result_num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            result_boxes = np.squeeze(result_boxes)
            result_classes = np.squeeze(result_classes)
            result_scores = np.squeeze(result_scores)
            
            for index, score in enumerate(result_scores):
                # Accepts when the confidence of detection 
                # is more than 50%
                if score < 0.5:
                    continue
                
                label = category_index[result_classes[index]]['name']
                
                # The result boxes returns coords of the detection
                # in format of ymin, xmin, ymax, xmax
                ymin, xmin, ymax, xmax = result_boxes[index]

                # Calculate the crop area depending
                # on the result coords above.
                
                # Geospatially width and height of 
                # detected area
                img_geo_width = abs(xmax - xmin)
                img_geo_height = abs(ymax - ymin)

                # Width and height in pixel of the 
                # detected image area
                pixel_width = width * img_geo_width
                pixel_height = height * img_geo_height

                x_min = int(abs(0 + xmin) * width)
                x_max = int(x_min + pixel_width)
                y_min = int(abs(0 + ymin) * height)
                y_max = int(y_min + pixel_height)

                # Crop the image on numpy array
                output_image = image_np[y_min:y_max, x_min:x_max]
                
                # only needed when we use load_image_into_numpy_array to 
                # load the image in np array
                # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_path, output_image)

                print(label, score)
                print()
