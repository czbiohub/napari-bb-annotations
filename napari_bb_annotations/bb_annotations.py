import argparse
import datetime
import glob
import logging
import os
import subprocess

from natsort import natsorted
import napari
import numpy as np
from skimage.io import imread

from gui import connect_to_viewer


# set up the annotation values and text display properties
# create the GUI for selecting the values
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create file handler which logs even info messages
now = datetime.datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
fh = logging.FileHandler("bb_annotations_{}.log".format(date_time), "w")
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
formatter = logging.Formatter(
    "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

IMAGE_FORMATS = [".png", ".tiff", ".tif", ".jpg", ".jpeg"]


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        logger.info("Path {} already exists, might be overwriting data".format(path))


def imread_convert(f):
    return imread(f).astype(np.uint8)


def add_image_shape_to_viewer(viewer, image, box_annotations, metadata):
    logger.info("adding image shape to viewer")
    logger.info("annotations added are {}".format(box_annotations))
    viewer.add_image(image, name="image", metadata=metadata)
    shapes = viewer.add_shapes(
        face_color='black', properties={'box_label': box_annotations}, ndim=3)

    shapes.text = 'box_label'
    # shapes.opacity = 0.2
    shapes.mode = 'add_rectangle'


def launch_viewer(path=None, format_of_files=None, box_annotations=None):
    with napari.gui_qt():
        viewer = napari.Viewer()
        if (path and format_of_files and box_annotations) is not None:
            connect_to_viewer(viewer, path, format_of_files, box_annotations)
        else:
            path, format_of_files, box_annotations = connect_to_viewer(viewer)
        assert os.path.exists(path)
        if type(box_annotations) is str:
            box_annotations = box_annotations.split(",")
        dirname = location = os.path.dirname(path)
        save_overlay_path = os.path.abspath(os.path.join(dirname, "overlay_dir"))
        create_dir_if_not_exists(save_overlay_path)

        if format_of_files not in IMAGE_FORMATS:
            filename_wo_format = os.path.basename(path).split(".")[0]
            output_frames_path = os.path.join(location, "frames_{}".format(filename_wo_format))
            create_dir_if_not_exists(output_frames_path)
            subprocess.check_call(
                'ffmpeg -i "{}" -f image2 "{}/video-frame%05d.jpg"'.format(
                    path, output_frames_path), shell=True)
            path = output_frames_path
            format_of_files = ".jpg"
        all_files = natsorted(
            glob.glob(os.path.join(path, "*" + format_of_files)))
        shape = imread(all_files[0]).shape
        total_files = len(all_files)
        if total_files == 0:
            logger.error("Exiting, no files left to annotate")
        chunk_size = 100
        pool_lists = []
        subset_stacks = []
        for i in range(0, len(all_files), chunk_size + 1):
            pool_list = all_files[i: i + chunk_size + 1]
            pool_lists.append(pool_list)
            if len(shape) == 3:
                stack = np.zeros((len(pool_list), shape[0], shape[1], shape[2]))
            else:
                stack = np.zeros((len(pool_list), shape[0], shape[1]))
            subset_stacks.append(stack)
        for index, pool_list in enumerate(pool_lists):
            subset_stack = subset_stacks[index]
            logger.info("subset_stack shape is {}".format(subset_stack.shape))
            metadata = {
                "save_overlay_path": save_overlay_path,
                "all_files": pool_list}
            add_image_shape_to_viewer(
                viewer, subset_stack, box_annotations, metadata)


def main():

    parser = argparse.ArgumentParser(
        description="napari viewer for bounding box and labels annotationn"
    )
    parser.add_argument(
        "--path",
        help="Images/Video to annotate with bounding boxes, " +
             "Please make sure your data is in a separate directory where new files/folders" +
             " can be added by the program and no other same format images exist",
        required=True)
    parser.add_argument(
        "--format_of_files",
        help="Format of input, including .",
        required=True)
    parser.add_argument(
        "--box_annotations",
        nargs="*",
        help="Comma separated classes you want to annotate in the images",
        required=True
    )

    args = parser.parse_args()
    launch_viewer(
        args.path,
        args.format_of_files,
        args.box_annotations,
    )


if __name__ == "__main__":
    main()
