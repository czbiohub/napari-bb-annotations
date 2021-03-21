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
import pandas as pd
from PIL import Image
from PIL import ImageDraw

import detect
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]
LUMI_CSV_COLUMNS = [
    'image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label', 'prob']
DEFAULT_CONFIDENCE = 0.4
DEFAULT_INFERENCE_COUNT = 1
DEFAULT_IMAGE_FORMAT = ".jpg"


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

    Args:
    path: path to label file.
    encoding: label file encoding.
    Returns:
    Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
        pairs = [line.split(' ', maxsplit=1) for line in lines]
        return {int(index): label.strip() for index, label in pairs}
    else:
        return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(
                EDGETPU_SHARED_LIB,
                {'device': device[0]} if device else {})])


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def detect_images(
        model, use_tpu, input_path, format_of_files,
        labels, threshold, output, count, overlaid):
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
    labels = load_labels(labels) if labels else {}
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
        interpreter = make_interpreter(model)
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
        image = Image.open(input_image)
        scale = detect.set_input(
            interpreter, image.size,
            lambda size: image.resize(size, Image.ANTIALIAS))

        for _ in range(count):
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_output(interpreter, threshold, scale)
            print('%.2f ms' % (inference_time * 1000))

        for obj in objs:
            df = df.append(
                {'image_id': input_image,
                 'xmin': obj.bbox.xmin,
                 'xmax': obj.bbox.xmax,
                 'ymin': obj.bbox.ymin,
                 'ymax': obj.bbox.ymax,
                 'label': labels.get(obj.id, obj.id),
                 'prob': obj.score}, ignore_index=True)

        if overlaid:
            image = image.convert('RGB')
            draw_objects(ImageDraw.Draw(image), objs, labels)
            image.save(
                os.path.join(
                    os.path.abspath(output),
                    os.path.basename(input_image)))
    df.to_csv(os.path.join(os.path.abspath(output), "bb_labels.csv"))


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
        help='Use Coral Edge TPU Accelerator to speed up detection',
        action='store_true')
    args = parser.parse_args()
    detect_images(
        args.model, args.edgetpu, args.input, args.format,
        args.labels, args.threshold, args.output, args.count, args.overlaid)

if __name__ == '__main__':
    main()
