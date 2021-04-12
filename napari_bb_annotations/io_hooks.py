"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import glob
import os
import subprocess

from natsort import natsorted
import numpy as np
from skimage.io import imread

from napari_plugin_engine import napari_hook_implementation
from napari_bb_annotations.constants_lumi import (
    BOX_ANNOTATIONS, MODEL, EDGETPU, IMAGE_FORMATS, EDGE_COLOR_CYCLE)


VIDEO_EXTS = [".avi", ".m4v", ".mkv", ".mp4"]


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


@napari_hook_implementation
def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    ext = os.path.splitext(path)[1].lower()
    print(ext)
    print(path)
    if ext not in VIDEO_EXTS or ext != "":
        return None
    print(path)

    assert os.path.exists(path)
    return reader_function


def convert_video(path):
    location = os.path.dirname(path)
    filename_wo_format = os.path.basename(path).split(".")[0]
    output_frames_path = os.path.join(
        location, "frames_{}".format(filename_wo_format))
    create_dir_if_not_exists(output_frames_path)
    subprocess.check_call(
        'ffmpeg -i "{}" -f image2 "{}/video-frame%05d.jpg"'.format(
            path, output_frames_path), shell=True)
    return output_frames_path


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.
    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari,
        and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    video_file = False
    for format_of_files in VIDEO_EXTS:
        if path.endswith(format_of_files):
            video_file = True
    print(video_file)
    if video_file:
        path = convert_video(path)
    path = path + os.sep if not path.endswith(os.sep) else path
    dirname = os.path.dirname(path)
    save_overlay_path = os.path.abspath(
        os.path.join(dirname, "overlay_dir"))
    all_files = []
    create_dir_if_not_exists(save_overlay_path)
    for format_of_files in IMAGE_FORMATS:
        format_of_files = format_of_files.lower()
        all_files.extend(natsorted(
            glob.glob(os.path.join(path, "*" + format_of_files))))

    # stack arrays into single array
    shape = imread(all_files[0]).shape
    total_files = len(all_files)
    if len(shape) == 3:
        stack = np.zeros(
            (total_files, shape[0], shape[1], shape[2]), dtype=np.uint8)
    else:
        stack = np.zeros((total_files, shape[0], shape[1]), dtype=np.uint8)
    for i in range(total_files):
        stack[i] = imread(all_files[i])

    layer_type = "image"  # optional, default is "image"
    num_files = len(all_files)
    metadata = {'metadata': {
        "save_overlay_path": save_overlay_path,
        "all_files": all_files,
        "box_annotations": BOX_ANNOTATIONS,
        "new_labels": [],
        "loaded": False,
        "updated": False,
        "tflite_inferenced": [False] * num_files,
        "threshold_inferenced": [False] * num_files,
        "lumi_inferenced": [False] * num_files,
        "model": MODEL,
        "edgetpu": EDGETPU},
        "name": "Image"}

    text_kwargs = {
        'text': '{box_label}: {unique_cell_id}%',
        'size': 8,
        'color': 'green'
    }
    add_kwargs = dict(
        face_color="black", edge_color='box_label',
        edge_color_cycle=EDGE_COLOR_CYCLE,
        properties={"box_label": BOX_ANNOTATIONS, "unique_cell_id": [0]},
        ndim=3,
        text=text_kwargs, name="Shapes", opacity=0.5)
    layer_list = [
        (stack, metadata, layer_type),
        (None, add_kwargs, "shapes")]
    print(layer_list)
    return layer_list
