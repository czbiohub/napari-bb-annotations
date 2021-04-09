import argparse
import importlib
import time
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
import detect
from skimage.draw import rectangle
from skimage.filters import threshold_otsu
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
DEFAULT_FILTER_AREA = 22680
DEFAULT_INFERENCE_COUNT = 1
DEFAULT_IMAGE_FORMAT = ".jpg"


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def out_of_bounds(xmin, xmax, ymin, ymax, org_width, org_height):
    if xmin >= org_width:
        xmin = org_width - 1

    if xmin < 0:
        xmin = 0

    if xmax >= org_width:
        xmax = org_width - 1

    if ymin >= org_height:
        ymin = org_height - 1

    if ymin < 0:
        ymin = 0

    if ymax >= org_height:
        ymax = org_height - 1
    return xmin, xmax, ymin, ymax


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
    model_file, *device = model_file.split('@')  # noqa
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


def check_if_bbox_not_background(bbox, thresholded_image):
    # Draw filled rectangle on the mask image
    mask = np.zeros_like(thresholded_image)
    rr, cc = rectangle(
        start=(bbox.xmin, bbox.ymin), end=(bbox.xmax, bbox.ymax))
    mask[cc, rr] = 1
    mask_nonzero_area = bbox.area
    intersection = np.logical_and(thresholded_image, mask).astype(np.uint8)
    bbox_is_in_foreground = True
    if (intersection.sum() / mask_nonzero_area) < 0.4:
        bbox_is_in_foreground = False
    return bbox_is_in_foreground
