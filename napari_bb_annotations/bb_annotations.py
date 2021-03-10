import napari
import os
import imageio
import tempfile
import shutil
import numpy as np
import logging
import skimage.color
from skimage.io import imread, imsave, ImageCollection
import datetime
from magicgui.widgets import ComboBox
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
    for bboxes in bbox:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def save_annotations_w_image(
    image,
    bboxes,
    labels,
    save_bb_labels_path,
    save_overlay_path,
    file_path,
):
    save_overlay_path = os.path.abspath(save_overlay_path)
    save_bb_labels_path = os.path.abspath(save_bb_labels_path)

    shape = image.shape
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
    logger.info("saving bboxes to {}".format(label_save_name))
    logger.info("saving images to {}".format(overlaid_save_name))

    image.save(overlaid_save_name)
    df.to_csv(os.path.join(
        save_bb_labels_path, "{}_preds_val.csv".format(filename_wo_format)))


def remove_tmp_dir(tempdir):
    try:
        shutil.rmtree(tempdir, ignore_errors=True)
    except OSError:
        pass


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
        properties={'box_label': box_annotations}, name="shape")
    shapes.text = 'box_label'
    shapes.opacity = 0.3
    shapes.mode = 'add_rectangle'


def launch_viewer():
    with napari.gui_qt():
        viewer = napari.Viewer()
        path, format_of_files, box_annotations = connect_to_viewer(viewer)
        label_menu = ComboBox(label='text label', choices=box_annotations)

        if format_of_files not in IMAGE_FORMATS:
            location = tempfile.mkdtemp(prefix='bb_annotations_')
            index = 0
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
        current_file = 0
        if len(image_collection) == 0:
            logger.error("Exiting, no files left to annotate")

        add_image_shape_to_viewer(
            viewer, image_collection[current_file], box_annotations
        )
        if len(image_collection) == current_file:
            logger.error("Exiting, no files left to annotate")

        @viewer.bind_key("Right")
        def next_image(viewer):
            # pop off old layers before adding new ones
            for i in reversed(range(len(viewer.layers))):
                viewer.layers.pop(i)

            nonlocal current_file
            current_file += 1
            try:
                add_image_shape_to_viewer(
                    viewer, image_collection[current_file], box_annotations
                )
            except IndexError:
                logger.error("Exiting, no more files left to annotate")
                remove_tmp_dir(location)
                sys.exit(1)
            logger.info(
                "Pressed next key {}".format(current_file))

        @viewer.bind_key("Left")
        def previous_image(viewer):
            # pop off old layers before adding new ones
            for i in reversed(range(len(viewer.layers))):
                viewer.layers.pop(i)

            nonlocal current_file
            current_file -= 1
            try:
                add_image_shape_to_viewer(
                    viewer, image_collection[current_file], box_annotations
                )
            except IndexError:
                logger.error("Exiting, no more files left to annotate")
                remove_tmp_dir(location)
                sys.exit(1)
            logger.info(
                "Pressed previous key{}".format(current_file)
            )

        @viewer.bind_key("s")
        def save_annotations(viewer):
            image = viewer.layers["image"]
            shape = viewer.layers["shape"]
            nonlocal current_file
            dirname = os.path.dirname(all_files[current_file])
            save_annotations_w_image(
                image.data,
                shape.data,
                shape.properties,
                os.path.join(dirname, save_bb_labels_dir),
                os.path.join(dirname, save_overlay_dir),
                all_files[current_file],
            )

        @viewer.bind_key("u")
        def update_annotations(viewer):
            label_property = text_property = "box_label"
            text_color = 'black'
            size = 10
            shapes_layer = viewer.layers['shape']

            def update_label_menu(event):
                """This is a callback function that updates the label menu when
                the current properties of the Shapes layer change
                """
                new_label = str(shapes_layer.current_properties[label_property][0])
                if new_label != label_menu.value:
                    label_menu.value = new_label

            shapes_layer.events.current_properties.connect(update_label_menu)

            def label_changed(event):
                """This is acallback that update the current properties on the Shapes layer
                when the label menu selection changes
                """
                selected_label = event.value
                current_properties = shapes_layer.current_properties
                current_properties[label_property] = np.asarray([selected_label])
                shapes_layer.current_properties = current_properties

            label_menu.changed.connect(label_changed)

            # To fix bug in currently have for creating emtpy layers with text
            # see: https://github.com/napari/napari/issues/2115
            def on_data(event):
                if viewer.layers['shape'].text.mode == 'none':
                    viewer.layers['shape'].text = text_property
                    viewer.layers['shape'].text.color = text_color
                    viewer.layers['shape'].text.size = size
            viewer.layers['shape'].events.set_data.connect(on_data)


def main():
    launch_viewer()


if __name__ == "__main__":
    main()
