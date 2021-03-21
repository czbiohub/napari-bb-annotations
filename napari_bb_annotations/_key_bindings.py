from magicgui.widgets import ComboBox, Container
import logging
import sys
import os
import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from napari import Viewer

from napari_bb_annotations.run_inference import detect_images, DEFAULT_CONFIDENCE, DEFAULT_INFERENCE_COUNT

LUMI_CSV_COLUMNS = [
    'image_id', 'xmin', 'xmax', 'ymin', 'ymax', 'label', 'prob']
# set up the annotation values and text display properties
# create the GUI for selecting the values
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create file handler which logs even info messages
now = datetime.datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
if not logger.handlers:
    logger.addHandler(handler)


def check_bbox(bbox, width, height):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    if xmin > width:
        xmin = width
    if xmin < 0:
        xmin = 0

    if xmax > width:
        xmax = width

    if ymin > height:
        ymin = height

    if ymin < 0:
        ymin = 0

    if ymax > height:
        ymax = height
    return np.array([xmin, ymin, xmax, ymax])


def draw_objects(draw, bboxes, labels):
    """Draws the bounding box and label for each object."""
    for index, bbox in enumerate(bboxes):
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                       outline='red')
        draw.text((bbox[0] + 10, bbox[1] + 10),
                  '%s' % (labels[index]),
                  fill='red')


def get_bbox_obj_detection(bbox):
    """
    Get the coordinates of the 4 corners of a
    bounding box - expected to be in 'xyxy' format.
    Result can be put directly into a napari shapes layer.
    Order: top-left, bottom-left, bottom-right, top-right
    numpy style [y, x]
    """
    if bbox.shape[1] == 3:
        x = (bbox[:, 2])
        y = (bbox[:, 1])
    else:
        x = (bbox[:, 1])
        y = (bbox[:, 0])

    x1 = x.min()
    y1 = y.min()
    x2 = x.max()
    y2 = y.max()

    return [x1, y1, x2, y2]


def create_label_menu(shapes_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters:
    -----------
    shapes_layer : napari.layers.Shapes
        a napari shapes layer
    labels : List[str]
        list of the possible text labels values.

    Returns:
    --------
    label_widget : magicgui.widgets.Container
        the container widget with the label combobox
    """
    # Create the label selection menu
    label_property = "box_label"
    label_menu = ComboBox(label='text label', choices=labels)
    label_widget = Container(widgets=[label_menu])

    def update_label_menu(event):
        """This is a callback function that updates the label menu when
        the current properties of the Shapes layer change
        """
        new_label = str(shapes_layer.current_properties[label_property][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    shapes_layer.events.current_properties.connect(update_label_menu)

    def label_changed(event):
        """This is a callback that update the current properties on the Shapes layer
        when the label menu selection changes
        """
        selected_label = event.value
        current_properties = shapes_layer.current_properties
        current_properties[label_property] = np.asarray([selected_label])
        shapes_layer.current_properties = current_properties

    label_menu.changed.connect(label_changed)

    return label_widget


@Viewer.bind_key('Shift-S')
def save_bb_labels(viewer):
    logger.info("Pressed key Shift-S")
    shape = viewer.layers["Shapes"]
    image = viewer.layers["image"]
    metadata = viewer.layers["image"].metadata
    stack = image.data
    bboxes = shape.data
    labels = shape.properties["box_label"].tolist()
    save_overlay_path = metadata["save_overlay_path"]
    csv_path = os.path.join(
        os.path.dirname(save_overlay_path), "bb_labels.csv")
    df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
    z_indices = []
    for bbox in bboxes:
        z_index = np.unique((bbox[:, 0])).tolist()
        assert len(z_index) == 1
        z_indices.append(z_index[0])
    z_indices = np.unique(z_indices).tolist()
    logger.info("z_indices {}".format(z_indices))
    logger.info("stack.shape {}".format(stack.shape))
    for stack_index in z_indices:
        # visualization image
        image_at_index = stack[int(stack_index)]
        file_path = metadata["all_files"][int(stack_index)]
        height, width = image_at_index.shape[:2]
        image_at_index = Image.fromarray(image_at_index)
        image_at_index = image_at_index.convert('RGB')
        bboxes_converted = []
        labels_for_image = []
        for index, bbox in enumerate(bboxes):
            z_index = np.unique((bbox[:, 0])).tolist()
            assert len(z_index) == 1
            if z_index[0] == stack_index:
                bbox = check_bbox(get_bbox_obj_detection(bbox), width, height)
                label = labels[index]
                df = df.append(
                    {'image_id': file_path,
                     'xmin': int(bbox[0]),
                     'ymin': int(bbox[1]),
                     'xmax': int(bbox[2]),
                     'ymax': int(bbox[3]),
                     'label': label}, ignore_index=True)
                labels_for_image.append(label)
                bboxes_converted.append(bbox)
        if len(bboxes_converted) != 0:
            draw_objects(
                ImageDraw.Draw(image_at_index),
                bboxes_converted, labels_for_image)
            # save images
            filename_wo_format = os.path.basename(file_path).split(".")[0]
            overlaid_save_name = os.path.join(
                save_overlay_path,
                "{}_overlaid.png".format(filename_wo_format)
            )
            logger.info("saving images to {}".format(overlaid_save_name))

            image_at_index.save(overlaid_save_name)
    logger.info("csv path is {}".format(csv_path))
    df.to_csv(csv_path)


@Viewer.bind_key('Shift-l')
def load_bb_labels(viewer):
    logger.info("Pressed key Shift-l")
    all_files = viewer.layers["image"].metadata["all_files"]
    dirname = os.path.dirname(all_files[0])
    df = pd.read_csv(os.path.join(
        dirname, "bb_labels.csv"))
    shape = viewer.layers["Shapes"]
    bboxes = shape.data
    labels = shape.properties["box_label"].tolist()
    logger.info("labels before {}".format(labels))
    for index, row in df.iterrows():
        x1 = row.xmin
        x2 = row.xmax
        y1 = row.ymin
        y2 = row.ymax
        label = row.label
        image_id = row.image_id
        z = all_files.index(image_id)
        bbox_rect = np.array(
            [[z, y1, x1], [z, y2, x1], [z, y2, x2], [z, y1, x2]]
        )
        bboxes.append(bbox_rect)
        labels.append(label)
    logger.info("labels after {}".format(labels))
    viewer.layers["Shapes"].data = bboxes
    viewer.layers["Shapes"].properties["box_label"] = np.array(labels)


@Viewer.bind_key('Shift-i')
def run_inference_on_images(viewer):
    logger.info("Pressed key Shift-i")
    all_files = viewer.layers["image"].metadata["all_files"]
    filename = all_files[0]
    dirname = os.path.dirname(filename)

    box_annotations = viewer.layers["image"].metadata["box_annotations"]
    model = viewer.layers["image"].metadata["model"]
    use_tpu = viewer.layers["image"].metadata["edgetpu"]

    labels_txt = os.path.join(dirname, "labels.txt")
    with open(labels_txt, 'w') as f:
        for index, label in enumerate(box_annotations):
            f.write("{} {}\n".format(index, label))

    format_of_files = os.path.splitext(filename)[1]
    detect_images(
        model, use_tpu, dirname, format_of_files,
        labels_txt, DEFAULT_CONFIDENCE, dirname, DEFAULT_INFERENCE_COUNT, False)


def update_layers(viewer, box_annotations):
    logger.info("Pressed update layers button")
    shapes_layer = viewer.layers['Shapes']

    label_widget = create_label_menu(shapes_layer, box_annotations)
    text_property = "box_label"
    text_color = 'green'
    # this is a hack to get around a bug we currently have
    # for creating emtpy layers with text
    # see: https://github.com/napari/napari/issues/2115

    def on_data(event):
        if shapes_layer.text.mode == 'none':
            shapes_layer.text = text_property
            shapes_layer.text.color = text_color
    shapes_layer.events.set_data.connect(on_data)
    # add the label selection gui to the viewer as a dock widget
    viewer.window.add_dock_widget(label_widget, area='right')
