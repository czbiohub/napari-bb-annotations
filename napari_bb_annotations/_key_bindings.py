import datetime
import logging
import os
import pickle
from typing import List

import luminoth.predict
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, Container, Table
from PIL import Image, ImageDraw
from .constants_lumi import BOX_ANNOTATIONS, LUMI_CSV_COLUMNS
from napari_bb_annotations.run_inference import (
    detect_images, DEFAULT_CONFIDENCE, DEFAULT_INFERENCE_COUNT)
from napari.utils.notifications import (
    Notification,
    notification_manager,
    show_info,
)
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

import centroid_tracker

MAX_DISAPPEARED_FRAMES = 35


def pickle_save(path, metadata_dct):
    with open(path, "wb") as fh:
        pickle.dump(metadata_dct, fh)


def pickle_load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# create the GUI for selecting the values
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create file handler which logs even info messages
now = datetime.datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


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


def draw_objects(draw, bboxes, labels, cell_ids):
    """Draws the bounding box and label for each object."""
    for index, bbox in enumerate(bboxes):
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                       outline='red')
        label = labels[index]
        if " " not in label:
            draw.text((bbox[0] + 10, bbox[1] + 10),
                      '{}-cell{}'.format(labels[index], cell_ids[index]),
                      fill='red')
        else:
            split1, split2 = label.split(" ")
            draw.text((bbox[0] + 10, bbox[1] + 10),
                      '{}/n{}-cell{}'.format(split1, split2, cell_ids[index]),
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


def create_label_menu(shapes_layer, image_layer):
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
    label_menu = ComboBox(label='text label', choices=BOX_ANNOTATIONS)
    label_widget = Container(widgets=[label_menu])

    def update_label_menu(event):
        """This is a callback function that updates the label menu when
        the current properties of the Shapes layer change
        """
        new_label = str(shapes_layer.current_properties[label_property][0])
        if new_label != label_menu.value and new_label in BOX_ANNOTATIONS:
            label_menu.value = new_label
        elif (new_label in image_layer.metadata["new_labels"] and
              new_label not in BOX_ANNOTATIONS and
              new_label not in label_menu.choices):
            label_menu.set_choice(new_label, new_label)

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


def update_summary_table(shapes_layer, image_layer):
    box_labels = shapes_layer.properties["box_label"].tolist()
    split_dict = {
        "data": [[box_labels.count(label)] for label in BOX_ANNOTATIONS],
        "index": tuple(BOX_ANNOTATIONS),
        "columns": ("c"),
    }
    table_widget = Table(value=split_dict)
    label_property = "box_label"
    label_menu = ComboBox(label='text label', choices=BOX_ANNOTATIONS)

    def update_table_on_label_change(event):
        """This is a callback function that updates the summary table when
        the current properties of the Shapes layer change or when you are
        moving to next image or at the end of stack
        """
        new_label = str(shapes_layer.current_properties[label_property][0])
        if new_label != label_menu.value:
            box_labels = shapes_layer.properties["box_label"].tolist()
            if new_label not in BOX_ANNOTATIONS:
                new_labels = image_layer.metadata["new_labels"]
                new_labels.append(new_label)
                image_layer.metadata["new_labels"] = new_labels
                index = sorted(BOX_ANNOTATIONS + [new_label])
                data = [[box_labels.count(label)] for label in index]
                split_dict = {
                    "data": data,
                    "index": tuple(index),
                    "columns": ("c"),
                }
            else:
                unique_labels = np.unique(
                    shapes_layer.properties['box_label']).tolist()
                data = [[box_labels.count(label)] for label in unique_labels]
                split_dict = {
                    "data": data,
                    "index": tuple(unique_labels),
                    "columns": ("c"),
                }
            table_widget.value = split_dict
    shapes_layer.events.current_properties.connect(
        update_table_on_label_change)

    def update_table_on_coordinates_change(event):
        unique_labels = np.unique(
            shapes_layer.properties['box_label']).tolist()
        box_labels = shapes_layer.properties["box_label"].tolist()
        data = [[box_labels.count(label)] for label in unique_labels]
        split_dict = {
            "data": data,
            "index": tuple(unique_labels),
            "columns": ("c"),
        }
        table_widget.value = split_dict

    shapes_layer.events.set_data.connect(update_table_on_coordinates_change)

    return table_widget


def save_bb_labels(viewer):
    logger.info("Pressed save bounding boxes, labels button")
    # TODO add unique_cell_id column while saving maybe?
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed save bounding boxes, labels button')
    shape = viewer.layers["Shapes"]
    image = viewer.layers["Image"]
    metadata = viewer.layers["Image"].metadata
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
                     'label': label,
                     }, ignore_index=True)
                labels_for_image.append(label)
                bboxes_converted.append(bbox)
        if len(bboxes_converted) != 0:
            draw_objects(
                ImageDraw.Draw(image_at_index),
                bboxes_converted, labels_for_image)
            # save images
            basename = os.path.basename(file_path)
            overlaid_save_name = os.path.join(
                save_overlay_path,
                "pred_{}".format(basename)
            )
            image_at_index.save(overlaid_save_name)
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info("csv path and overlaid images path is {}".format(csv_path, save_overlay_path))
    logger.info("csv path is {}".format(csv_path))
    df.to_csv(csv_path)


def load_bb_labels(viewer):
    logger.info("Pressed load bounding box, labels button")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)

        show_info('Pressed load bounding box, labels button')
    if viewer.layers["Image"].metadata["loaded"]:
        return
    all_files = viewer.layers["Image"].metadata["all_files"]
    dirname = os.path.dirname(all_files[0])
    csv_path = os.path.join(dirname, "bb_labels.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        shape = viewer.layers["Shapes"]
        bboxes = shape.data
        labels = shape.properties["box_label"].tolist()
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
        shapes_layer = viewer.layers["Shapes"]
        shapes_layer.data = bboxes
        viewer.layers["Image"].metadata["loaded"] = True
        shapes_layer.properties["box_label"] = np.array(labels)
        shapes_layer.text.refresh_text(shapes_layer.properties)
    update_layers(viewer)


def run_inference_on_images(viewer):
    logger.info("Pressed button for running prediction")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)

        show_info('Pressed button for running prediction, takes up to 1s per image')
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
        labels_txt, DEFAULT_CONFIDENCE, dirname,
        DEFAULT_INFERENCE_COUNT, False)


def run_segmentation_on_images(viewer):
    logger.info("Pressed button for running segmentation")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)

        show_info('Pressed button for running segmentation')
    # label image regions
    all_files = viewer.layers["image"].metadata["all_files"]
    for path in all_files:
        numpy_image = imread(path)[:, :, 0]
        thresholded_image = np.zeros(
            (numpy_image.shape[0], numpy_image.shape[1]), dtype=np.uint8)
        thresh_value = threshold_otsu(numpy_image)
        thresholded_image[numpy_image < thresh_value] = 255
        label_image = label(thresholded_image)
        df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)
        height, width = numpy_image.shape[:2]

        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 500:
                # draw rectangle around segmented coins
                xmin, ymin, xmax, ymax = region.bbox
                bbox = check_bbox(xmin, ymin, xmax, ymax, width, height)
                label = labels[index]
                df = df.append(
                    {'image_id': file_path,
                     'xmin': int(bbox[0]),
                     'ymin': int(bbox[1]),
                     'xmax': int(bbox[2]),
                     'ymax': int(bbox[3]),
                     'label': "healthy",
                     }, ignore_index=True)
                labels_for_image.append(label)
                bboxes_converted.append(bbox)


def run_tracking_on_images(viewer):
    logger.info("Pressed button for running tracking")
    # TODO Update summary table based on tracking
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)

        show_info('Pressed button for running tracking')
    all_files = viewer.layers["image"].metadata["all_files"]

    df = centroid_tracker.df_centroid_tracking_rectangles(
        df, MAX_DISAPPEARED_FRAMES, all_files)


def update_layers(viewer):
    logger.info("Updating layers")
    if viewer.layers["Image"].metadata["updated"]:
        return
    shapes_layer = viewer.layers['Shapes']
    image_layer = viewer.layers['Image']
    shapes_layer.mode = 'add_rectangle'

    label_widget = create_label_menu(shapes_layer, image_layer)
    # add the label selection gui to the viewer as a dock widget
    viewer.window.add_dock_widget(label_widget, area='right')
    table_widget = update_summary_table(
        viewer.layers["Shapes"],
        viewer.layers["Image"])
    table_widget.min_width = 300
    table_widget.min_height = 400
    table_widget.max_width = 300
    table_widget.max_height = 400
    viewer.layers["Image"].metadata["updated"] = True
    viewer.window.add_dock_widget(table_widget, area='right')


def get_properties_table(current_properties):
    split_dict = {
        "data": [current_properties],
        "index": tuple(["properties"]),
        "columns": ("c"),
    }
    table_widget = Table(value=split_dict)
    return table_widget


def edit_bb_labels(viewer):
    logger.info("Pressed edit labels for a bounding box button")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed edit bounding box label button')
    shapes_layer = viewer.layers["Shapes"]
    shapes_layer.mode = 'select'

    current_properties = shapes_layer.current_properties['box_label'].tolist()
    table_widget = get_properties_table(current_properties)

    def on_item_changed(item):
        # item will be an instance of `QTableWidgetItem`
        # https://doc.qt.io/Qt-5/qtablewidgetitem.html
        # whatever you want to do with the new value
        new_label = item.data(table_widget._widget._DATA_ROLE)
        new_label = new_label.capitalize()
        current_properties = shapes_layer.current_properties
        current_properties['box_label'] = np.asarray([new_label])
        shapes_layer.current_properties = current_properties
        table_widget.clear()
        # set the shapes layer mode back to pan/zoom
        shapes_layer.mode = 'add_rectangle'

    table_widget.native.itemChanged.connect(on_item_changed)

    viewer.window.add_dock_widget(table_widget, area='left')


def load_bb_labels_for_image(viewer):
    logger.info("Loading inference results for image")
    all_files = viewer.layers["Image"].metadata["all_files"]
    index_of_image = viewer.dims.current_step[0]
    filename = all_files[index_of_image]
    dirname = os.path.dirname(all_files[0])
    df = pd.read_csv(os.path.join(dirname, "bb_labels.csv"), index_col=False)
    # Filter out the df for all the bounding boxes in one image
    tmp_df = df[df.image_id == filename]
    shape = viewer.layers["Shapes"]
    bboxes = shape.data
    labels = shape.properties["box_label"].tolist()
    for index, row in tmp_df.iterrows():
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
    shapes_layer = viewer.layers["Shapes"]
    shapes_layer.data = bboxes
    shapes_layer.properties["box_label"] = np.array(labels)
    shapes_layer.text.refresh_text(shapes_layer.properties)
