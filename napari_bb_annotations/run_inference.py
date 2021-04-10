# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects in a given image."""

import argparse
import importlib
import time
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import napari_bb_annotations.detect as detect
import napari_bb_annotations.utils as utils

LUMI_CSV_COLUMNS = [
    'image_id', 'xmin', 'xmax', 'ymin',
    'ymax', 'label', 'prob', 'unique_cell_id']
DEFAULT_CONFIDENCE = 0.4
DEFAULT_FILTER_AREA = 22680
DEFAULT_INFERENCE_COUNT = 1
DEFAULT_IMAGE_FORMAT = ".jpg"


def detect_images(
        model, use_tpu, input_path, format_of_files,
        labels, threshold, output, count, overlaid,
        area_filter, filter_background_bboxes):
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
    output = os.path.abspath(output)
    utils.create_dir_if_not_exists(output)
    labels = utils.load_labels(labels) if labels else {}
    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime,
    # else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter
    if use_tpu:
        interpreter = utils.make_interpreter(model)
    else:
        interpreter = Interpreter(model_path=model)

    interpreter.allocate_tensors()

    input_images = []
    if input_path.endswith(format_of_files):
        input_images.append(input_path)
    else:
        for input_image in glob.glob(
                os.path.join(input_path, "*" + format_of_files)):
            input_images.append(input_image)
    print('----INFERENCE TIME----')
    if use_tpu:
        print('Note: The first inference is slow because it includes',
              'loading the model into Edge TPU memory.')
    for input_image in input_images:
        print(input_image)
        image = Image.open(input_image)
        scale = detect.set_input(
            interpreter, image.size,
            lambda size: image.resize(size, Image.ANTIALIAS))
        numpy_image = np.array(image)
        if filter_background_bboxes:
            numpy_image = numpy_image[:, :, 0]
            thresholded_image = np.zeros(
                (numpy_image.shape[0], numpy_image.shape[1]), dtype=np.uint8)
            thresh_value = 128
            thresholded_image[numpy_image < thresh_value] = 1
        for _ in range(count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_output(interpreter, threshold, scale)
            print('%.2f ms' % (inference_time * 1000))
        filtered_objs = []
        for obj in objs:
            xmin, xmax, ymin, ymax = \
                obj.bbox.xmin, obj.bbox.xmax, obj.bbox.ymin, obj.bbox.ymax
            org_height, org_width = numpy_image.shape[:2]
            xmin, xmax, ymin, ymax = utils.out_of_bounds(
                xmin, xmax, ymin, ymax, org_width, org_height)
            bbox = detect.BBox(xmin, ymin, xmax, ymax)
            if obj.bbox.area < area_filter:
                if filter_background_bboxes:
                    if utils.check_if_bbox_not_background(
                            bbox, thresholded_image):
                        df = df.append(
                            {'image_id': input_image,
                             'xmin': xmin,
                             'xmax': xmax,
                             'ymin': ymin,
                             'ymax': ymax,
                             'label': labels.get(obj.id, obj.id),
                             'prob': obj.score}, ignore_index=True)
                        filtered_objs.append(obj)
                else:
                    df = df.append(
                        {'image_id': input_image,
                         'xmin': xmin,
                         'xmax': xmax,
                         'ymin': ymin,
                         'ymax': ymax,
                         'label': labels.get(obj.id, obj.id),
                         'prob': obj.score}, ignore_index=True)
                    filtered_objs.append(obj)
        print(len(filtered_objs))
        if overlaid:
            image = image.convert('RGB')
            utils.draw_objects(ImageDraw.Draw(image), filtered_objs, labels)
            image.save(
                os.path.join(
                    output,
                    os.path.basename(input_image)))
    df['unique_cell_id'].values[:] = 0
    df.to_csv(os.path.join(output, "bb_labels.csv"))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True,
        help='File path of .tflite file.')
    parser.add_argument(
        '-i', '--input', required=True,
        help='File path of image to process.')
    parser.add_argument(
        '-l', '--labels',
        help='File path of labels file.')
    parser.add_argument(
        '-t', '--threshold', type=float, default=DEFAULT_CONFIDENCE,
        help='Score threshold for detected objects.')
    parser.add_argument(
        '-o', '--output',
        help='File path for the result image with annotations ' +
        'and csv file containing bboxes, annotations')
    parser.add_argument(
        '-c', '--count', type=int, default=DEFAULT_INFERENCE_COUNT,
        help='Number of times to run inference')
    parser.add_argument(
        '-f', '--format', type=str, default=DEFAULT_IMAGE_FORMAT,
        help='Format of image')
    parser.add_argument(
        '--edgetpu',
        help='Use Coral Edge TPU Accelerator to speed up detection',
        action='store_true')
    parser.add_argument(
        '--overlaid',
        help='Enable overlaid',
        action='store_true')
    parser.add_argument(
        '--area_filter',
        help='Enable filtering bounding boxes of area', type=int,
        default=DEFAULT_FILTER_AREA)
    parser.add_argument(
        '--filter_background_bboxes',
        help='Enable filtering bounding boxes that are in the background',
        action='store_true')
    args = parser.parse_args()
    detect_images(
        args.model, args.edgetpu, args.input, args.format,
        args.labels, args.threshold, args.output, args.count, args.overlaid,
        args.area_filter, args.filter_background_bboxes)

if __name__ == '__main__':
    main()
