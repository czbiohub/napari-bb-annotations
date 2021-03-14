import datetime
import logging
import os
import subprocess

import napari
import numpy as np
import pandas as pd
from skimage.io import imread, ImageCollection, collection

from gui import connect_to_viewer
from PIL import ImageDraw


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
LUMI_CSV_COLUMNS = [
    'image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label', 'prob']


def draw_objects(draw, bboxes, labels):
    """Draws the bounding box and label for each object."""
    for index, bbox in enumerate(bboxes):
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s' % (labels[index]),
                  fill='red')


def save_annotations_w_image(
    stack,
    bboxes,
    labels,
    save_overlay_path,
    file_path,
):
    save_overlay_path = os.path.abspath(save_overlay_path)
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
    for image in stack:
        # visualization image
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), bboxes, labels)
        for index, bbox in bboxes:
            df = df.append(
                {'image_id': file_path,
                 'xmin': bbox.xmin,
                 'xmax': bbox.xmax,
                 'ymin': bbox.ymin,
                 'ymax': bbox.ymax,
                 'label': labels[index]}, ignore_index=True)

        # save images
        filename_wo_format = os.path.basename(file_path).split(".")[0]
        overlaid_save_name = os.path.join(
            save_overlay_path,
            "{}_overlaid.png".format(filename_wo_format)
        )
        logger.info("saving images to {}".format(overlaid_save_name))

        image.save(overlaid_save_name)
        df.to_csv(os.path.join(
            os.path.dirname(save_overlay_path), "{}_preds_val.csv".format(filename_wo_format)))


def get_bbox_obj_detection(bbox):
    """
    Get the coordinates of the 4 corners of a
    bounding box - expected to be in 'xyxy' format.
    Result can be put directly into a napari shapes layer.

    Order: top-left, bottom-left, bottom-right, top-right
    numpy style [y, x]

    """
    x = (bbox[:, 1])
    y = (bbox[:, 0])

    x1 = x.min()
    y1 = y.min()
    x2 = x.max()
    y2 = y.max()

    return np.array([x1, y1, x2, y2])


def imread_convert(f):
    return imread(f).astype(np.uint8)


def add_image_shape_to_viewer(viewer, image, box_annotations):
    logger.info("adding image shape to viewer")
    logger.info("annotations added are {}".format(box_annotations))
    viewer.add_image(image, name="image")
    shapes = viewer.add_shapes(
        face_color='black', properties={'box_label': box_annotations}, ndim=3)

    shapes.text = 'box_label'
    shapes.opacity = 0.3
    shapes.mode = 'add_rectangle'


def launch_viewer():
    with napari.gui_qt():
        viewer = napari.Viewer()
        path, format_of_files, box_annotations = connect_to_viewer(viewer)
        dirname = location = os.path.dirname(path)
        save_overlay_path = os.path.join(dirname, "overlay_dir"),

        if format_of_files not in IMAGE_FORMATS:
            subprocess.check_call(
                'ffmpeg -i "{}" -f image2 "{}/video-frame%05d.jpg"'.format(
                    path, location), shell=True)
            image_collection = ImageCollection(
                location + os.sep + "*.png",
                load_func=imread_convert)
        else:
            image_collection = ImageCollection(
                path + os.sep + "*" + format_of_files,
                load_func=imread_convert)
        stack = collection.concatenate_images(image_collection)
        all_files = image_collection.files
        logger.info("stack shape is {}".format(stack.shape))
        total_files = len(image_collection)
        if total_files == 0:
            logger.error("Exiting, no files left to annotate")

        add_image_shape_to_viewer(
            viewer, stack, box_annotations)

        @viewer.bind_key("s")
        def save_annotations(viewer):
            # TODO load frame here for both image and shapes here instead?
            shape = viewer.layers["shape"]
            image = viewer.layers["image"]
            current_file = shape.z_index
            save_annotations_w_image(
                image.data,
                shape.data,
                shape.properties,
                save_overlay_path,
                all_files[current_file],
            )


def main():
    launch_viewer()


if __name__ == "__main__":
    main()
