import csv
import datetime
import json
import logging
import functools
import os
import pickle
import subprocess
from typing import List

import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, Container, Table
from PIL import Image, ImageDraw
from napari_bb_annotations.constants_lumi import (
    BOX_ANNOTATIONS, LUMI_CSV_COLUMNS)
from napari_bb_annotations.run_inference import (
    detect_images, DEFAULT_INFERENCE_COUNT,
    DEFAULT_FILTER_AREA)
from napari.utils.notifications import (
    Notification,
    notification_manager,
    show_info,
)
from skimage.filters import threshold_otsu
import skimage.measure


@functools.lru_cache()
def get_predictor_network(checkpoint, max_detections, min_prob):
    from luminoth.utils.predicting import PredictorNetwork
    from luminoth.tools.checkpoint import get_checkpoint_config

    # Resolve the config to use and initialize the model.
    config = get_checkpoint_config(checkpoint)

    # Filter bounding boxes according to `min_prob` and `max_detections`.
    if config.model.type == 'fasterrcnn':
        if config.model.network.with_rcnn:
            config.model.rcnn.proposals.total_max_detections = max_detections
        else:
            config.model.rpn.proposals.post_nms_top_n = max_detections
        config.model.rcnn.proposals.min_prob_threshold = min_prob
    elif config.model.type == 'ssd':
        config.model.proposals.total_max_detections = max_detections
        config.model.proposals.min_prob_threshold = min_prob
    else:
        raise ValueError(
            "Model type '{}' not supported".format(config.model.type)
        )
    network = PredictorNetwork(config)
    return network


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
    error = False
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    if xmin > width:
        xmin = width
        error = True
    if xmin < 0:
        xmin = 0
        error = True
    if xmax > width:
        xmax = width
        error = True
    if ymin > height:
        ymin = height
        error = True
    if ymin < 0:
        ymin = 0
        error = True
    if ymax > height:
        ymax = height
        error = True
    return error, np.array([xmin, ymin, xmax, ymax])


def draw_objects(draw, bboxes, labels):
    """Draws the bounding box and label for each object."""
    for index, bbox in enumerate(bboxes):
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                       outline='red')
        label = labels[index]
        if " " not in label:
            draw.text((bbox[0] + 10, bbox[1] + 10),
                      '{}'.format(labels[index]),
                      fill='red')
        else:
            split1, split2 = label.split(" ")
            draw.text((bbox[0] + 10, bbox[1] + 10),
                      '{}/n{}'.format(split1, split2),
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
        elif (new_label not in BOX_ANNOTATIONS and
              new_label not in label_menu.choices):
            new_labels = image_layer.metadata["new_labels"]
            new_labels.append(new_label)
            new_labels = np.unique(new_labels).tolist()
            image_layer.metadata["new_labels"] = new_labels
            label_menu.set_choice(new_label, new_label)

    shapes_layer.events.current_properties.connect(update_label_menu)

    def label_changed(event):
        """This is a callback that update the current properties on the Shapes layer
        when the label menu selection changes
        """
        selected_label = event.value
        current_properties = shapes_layer.current_properties
        current_properties[label_property] = np.asarray([selected_label], dtype='<U32')
        shapes_layer.current_properties = current_properties
        shapes_layer.text.refresh_text(shapes_layer.properties)

    label_menu.changed.connect(label_changed)

    return label_widget


def update_summary_table(viewer):
    shapes_layer = viewer.layers["Shapes"]
    image_layer = viewer.layers["Image"]
    logger.info("Updating summary table")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Updating summary table')
    box_labels = shapes_layer.properties["box_label"].tolist()
    total_sum = len(box_labels)
    data = []
    new_labels = np.unique(box_labels).tolist()
    index = sorted(np.unique(new_labels + BOX_ANNOTATIONS).tolist())
    for label in index:
        count_label = box_labels.count(label)
        data.append([count_label, round((count_label * 100) / total_sum, 2)])
    split_dict = {
        "data": data,
        "index": tuple(index),
        "columns": ("c", "p"),
    }

    table_widget = Table(value=split_dict)
    return table_widget


def save_bb_labels(viewer):
    logger.info("Pressed save bounding boxes, labels button")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed save bounding boxes, labels button')
    shape = viewer.layers["Shapes"]
    image = viewer.layers["Image"]
    metadata = viewer.layers["Image"].metadata
    save_overlay_path = metadata["save_overlay_path"]
    stack = image.data
    bboxes = shape.data
    labels = shape.properties["box_label"].tolist()
    csv_path = os.path.join(
        os.path.dirname(save_overlay_path), "bb_labels.csv")

    z_indices = []
    for bbox in bboxes:
        z_index = np.unique((bbox[:, 0])).tolist()
        assert len(z_index) == 1
        z_indices.append(z_index[0])
    z_indices = np.unique(z_indices).tolist()
    height, width = stack[0].shape[:2]
    bb_labels_rows = []
    for stack_index in z_indices:
        # visualization image
        file_path = metadata["all_files"][int(stack_index)]
        for index, bbox in enumerate(bboxes):
            z_index = np.unique((bbox[:, 0])).tolist()
            assert len(z_index) == 1
            if z_index[0] == stack_index:
                bbox = get_bbox_obj_detection(bbox)
                if not check_bbox(bbox, width, height)[0]:
                    label = labels[index]
                    bb_labels_rows.append(
                        [file_path, int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3]), label, 0, 0])

    with open(csv_path, "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile, lineterminator="\n")

        # writing the fields
        csvwriter.writerow(LUMI_CSV_COLUMNS)

        # writing the data rows
        csvwriter.writerows(bb_labels_rows)
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info("csv path is {}".format(csv_path))
    logger.info("csv path is {}".format(csv_path))
    data = viewer.layers["Image"].metadata["table_widget"].value
    json_path = os.path.join(
        os.path.dirname(save_overlay_path), "summary_table.json")
    with open(json_path, 'w') as fp:
        json.dump(data, fp)


def save_overlaid(viewer):
    logger.info("Pressed save annotations overlay button")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed save annotations overlay button')
    shape = viewer.layers["Shapes"]
    image = viewer.layers["Image"]
    metadata = viewer.layers["Image"].metadata
    stack = image.data
    bboxes = shape.data
    labels = shape.properties["box_label"].tolist()
    save_overlay_path = metadata["save_overlay_path"]
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
                _, bbox = check_bbox(get_bbox_obj_detection(bbox), width, height)
                label = labels[index]
                labels_for_image.append(label)
                bboxes_converted.append(bbox)
        if len(bboxes_converted) != 0:
            draw_objects(
                ImageDraw.Draw(image_at_index),
                bboxes_converted,
                labels_for_image)
            # save images
            basename = os.path.basename(file_path)
            overlaid_save_name = os.path.join(
                save_overlay_path,
                "pred_{}".format(basename)
            )
            image_at_index.save(overlaid_save_name)


def load_bb_labels(viewer):
    logger.info("Pressed load bounding box, labels button")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed load bounding boxes, labels button')
    logger.info("Pressed load bounding box, labels button")
    if set(viewer.layers["Image"].metadata["loaded"]) == {True}:
        return
    all_files = viewer.layers["Image"].metadata["all_files"]
    dirname = os.path.dirname(all_files[0])
    csv_path = os.path.join(dirname, "bb_labels.csv")
    shapes_layer = viewer.layers["Shapes"]
    bboxes = []
    labels = []
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                line_count += 1
                x1 = row["xmin"]
                x2 = row["xmax"]
                y1 = row["ymin"]
                y2 = row["ymax"]
                label = row["label"]
                image_id = row["image_id"]
                z = all_files.index(image_id)
                bbox_rect = np.array(
                    [[z, y1, x1], [z, y2, x1], [z, y2, x2], [z, y1, x2]]
                )
                bboxes.append(bbox_rect)
                labels.append(label)
        viewer.layers["Image"].metadata["loaded"] = [True] * len(all_files)
        shapes_layer.data = bboxes
        shapes_layer.properties["box_label"] = np.array(labels, dtype='<U32')
        shapes_layer.text.refresh_text(shapes_layer.properties)
    table_widget = update_layers(viewer)
    box_labels = shapes_layer.properties['box_label'].tolist()
    index = sorted(np.unique(shapes_layer.properties['box_label']).tolist())
    index = sorted(list(set(index + BOX_ANNOTATIONS)))
    total_sum = len(box_labels)
    data = []
    for label in index:
        count_label = box_labels.count(label)
        data.append([count_label, round((count_label * 100) / total_sum, 2)])
    split_dict = {
        "data": data,
        "index": tuple(index),
        "columns": ("c", "p"),
    }
    table_widget.value = split_dict


def run_inference_on_images(viewer):
    logger.info("Pressed button for running prediction")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed button for running prediction using tflite model')
    all_files = viewer.layers["Image"].metadata["all_files"]
    filename = all_files[0]
    dirname = os.path.dirname(filename)
    inference_metadata_path = os.path.join(
        dirname, "inference_metadata.pickle")
    already_inferenced = [False] * len(all_files)
    if os.path.exists(inference_metadata_path):
        inference_metadata = pickle_load(inference_metadata_path)
        if "tflite_inferenced" in inference_metadata:
            already_inferenced = inference_metadata["tflite_inferenced"]
        if set(already_inferenced) == {True}:
            with notification_manager:
                # save all of the events that get emitted
                store: List[Notification] = []   # noqa
                _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
                notification_manager.notification_ready.connect(_append)
                show_info('Already ran tflite prediction, loading')
                load_bb_labels(viewer)
    if set(already_inferenced) == {False}:
        box_annotations = viewer.layers["Image"].metadata["box_annotations"]
        model = viewer.layers["Image"].metadata["tflite_model"]
        use_tpu = viewer.layers["Image"].metadata["edgetpu"]

        labels_txt = os.path.join(dirname, "labels.txt")
        with open(labels_txt, 'w') as f:
            for index, label in enumerate(box_annotations):
                f.write("{} {}\n".format(index, label))

        format_of_files = os.path.splitext(filename)[1]
        saved_model_path = os.path.join(dirname, "output_tflite_graph.tflite")
        if not os.path.exists(saved_model_path):
            subprocess.check_call(
                "curl {} --output {}".format(model, saved_model_path), shell=True)
        # TODO set lower intentionally for this release of the app
        confidence = 0.1
        detect_images(
            saved_model_path, use_tpu, dirname, format_of_files,
            labels_txt, confidence, dirname,
            DEFAULT_INFERENCE_COUNT, False,
            DEFAULT_FILTER_AREA, True)
        inferenced_list = [True] * len(all_files)
        viewer.layers["Image"].metadata["tflite_inferenced"] = inferenced_list
        viewer.layers["Image"].metadata["loaded"] = [False] * len(all_files)
        metadata = {"tflite_inferenced": inferenced_list}
        pickle_save(inference_metadata_path, metadata)
        load_bb_labels(viewer)


def run_segmentation_on_images(viewer):
    logger.info("Pressed button for running segmentation")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed button for running segmentation')
    # label image regions
    all_files = viewer.layers["Image"].metadata["all_files"]
    stack = viewer.layers["Image"].data
    dirname = os.path.dirname(all_files[0])
    inference_metadata_path = os.path.join(
        dirname, "inference_metadata.pickle")
    already_inferenced = [False] * len(all_files)
    if os.path.exists(inference_metadata_path):
        inference_metadata = pickle_load(inference_metadata_path)
        if "threshold_inferenced" in inference_metadata:
            already_inferenced = inference_metadata["threshold_inferenced"]
        if set(already_inferenced) == {True}:
            with notification_manager:
                # save all of the events that get emitted
                store: List[Notification] = []   # noqa
                _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
                notification_manager.notification_ready.connect(_append)
                show_info('Already ran threshold prediction, click load')
            logger.info("Already ran threshold prediction, click load")
    shape = stack.shape
    bb_labels_rows = []
    if set(already_inferenced) == {False}:
        for index, file_path in enumerate(all_files):
            if len(shape) == 4:
                numpy_image = stack[index][:, :, 0]
            elif len(shape) == 3:
                numpy_image = stack[index]
            height, width = numpy_image.shape[:2]
            thresholded_image = np.zeros(
                (numpy_image.shape[0], numpy_image.shape[1]), dtype=np.uint8)
            thresh_value = threshold_otsu(numpy_image)
            thresholded_image[numpy_image < thresh_value] = 255
            label_image = skimage.measure.label(thresholded_image)

            for region in skimage.measure.regionprops(label_image):
                # take regions with large enough areas
                if region.area >= 1000 and region.area <= 8000:
                    ymin, xmin, ymax, xmax = region.bbox
                    if not check_bbox([xmin, ymin, xmax, ymax], width, height)[0]:
                        bb_labels_rows.append([file_path, xmin, xmax, ymin, ymax, "healthy", 0, 0])
        inferenced_list = [True] * len(all_files)
        viewer.layers["Image"].metadata["threshold_inferenced"] = inferenced_list
        metadata = {"threshold_inferenced": inferenced_list}
        viewer.layers["Image"].metadata["loaded"] = [False] * len(all_files)
        pickle_save(inference_metadata_path, metadata)
        with open(os.path.join(dirname, "bb_labels.csv"), "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile, lineterminator="\n")

            # writing the fields
            csvwriter.writerow(LUMI_CSV_COLUMNS)

            # writing the data rows
            csvwriter.writerows(bb_labels_rows)
        load_bb_labels(viewer)


def update_layers(viewer):
    logger.info("Updating layers")
    if viewer.layers["Image"].metadata["updated"]:
        return viewer.layers["Image"].metadata["table_widget"]
    shapes_layer = viewer.layers['Shapes']
    image_layer = viewer.layers['Image']
    shapes_layer.mode = 'add_rectangle'

    label_widget = create_label_menu(shapes_layer, image_layer)
    # add the label selection gui to the viewer as a dock widget
    viewer.window.add_dock_widget(label_widget, area='right')
    table_widget = update_summary_table(viewer)
    table_widget.min_width = 300
    table_widget.min_height = 400
    table_widget.max_width = 300
    table_widget.max_height = 400
    viewer.layers["Image"].metadata["updated"] = True
    viewer.window.add_dock_widget(table_widget, area='right')
    viewer.layers["Image"].metadata["table_widget"] = table_widget
    return table_widget


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
        new_labels = viewer.layers["Image"].metadata["new_labels"]
        new_labels.append(new_label)
        new_labels = np.unique(new_labels).tolist()
        viewer.layers["Image"].metadata["new_labels"] = new_labels
        current_properties = shapes_layer.current_properties
        current_properties['box_label'] = np.asarray([new_label], dtype='<U32')
        shapes_layer.current_properties = current_properties
        table_widget.clear()
        table_widget.visible = False
        # set the shapes layer mode back to rectangle
        shapes_layer.mode = 'add_rectangle'

    table_widget.native.itemChanged.connect(on_item_changed)

    viewer.window.add_dock_widget(table_widget, area='left')


def load_bb_labels_for_image(viewer, csv_path):
    logger.info("Loading inference results for image")
    all_files = viewer.layers["Image"].metadata["all_files"]
    index_of_image = viewer.dims.current_step[0]
    if not viewer.layers["Image"].metadata["loaded"][index_of_image]:
        filename = all_files[index_of_image]
        dirname = os.path.dirname(all_files[0])
        df = pd.read_csv(csv_path, index_col=False)
        # Filter out the df for all the bounding boxes in one image
        tmp_df = df[df.image_id == filename]
        shapes_layer = viewer.layers["Shapes"]
        bboxes = shapes_layer.data
        labels = shapes_layer.properties["box_label"].tolist()
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
        shapes_layer.data = bboxes
        shapes_layer.properties["box_label"] = np.array(labels, dtype='<U32')
        shapes_layer.text.refresh_text(shapes_layer.properties)
        viewer.layers["Image"].metadata["loaded"][index_of_image] = True
    table_widget = update_layers(viewer)
    box_labels = shapes_layer.properties['box_label'].tolist()
    index = sorted(np.unique(shapes_layer.properties['box_label']).tolist())
    index = sorted(list(set(index + BOX_ANNOTATIONS)))
    total_sum = len(box_labels)
    data = []
    for label in index:
        count_label = box_labels.count(label)
        data.append([count_label, round((count_label * 100) / total_sum, 2)])
    split_dict = {
        "data": data,
        "index": tuple(index),
        "columns": ("c", "p"),
    }
    table_widget.value = split_dict


def run_lumi_on_image(viewer):
    logger.info("Pressed button to run luminoth prediction")
    with notification_manager:
        # save all of the events that get emitted
        store: List[Notification] = []   # noqa
        _append = lambda e: store.append(e)  # lambda needed on py3.7  # noqa
        notification_manager.notification_ready.connect(_append)
        show_info('Pressed button for running prediction using tensorflow model')
    image_layer = viewer.layers["Image"]
    metadata = image_layer.metadata
    all_files = metadata["all_files"]
    index_of_image = viewer.dims.current_step[0]
    filename = all_files[index_of_image]
    dirname = os.path.dirname(filename)
    model = image_layer.metadata["model"]
    inference_metadata_path = os.path.join(
        dirname, "inference_metadata.pickle")

    csv_path = os.path.join(dirname, "lumi_bb_labels.csv")
    inferenced_list = [False] * len(all_files)
    if os.path.exists(inference_metadata_path):
        inference_metadata = pickle_load(inference_metadata_path)
        if "lumi_inferenced" in inference_metadata:
            inferenced_list = inference_metadata["lumi_inferenced"]
            if inferenced_list[index_of_image]:
                load_bb_labels_for_image(viewer, csv_path)
                return
    if not viewer.layers["Image"].metadata["lumi_inferenced"][index_of_image]:
        # To not overwrite the existing csv, and lose the predictions per image
        # from last image
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=False)
        else:
            df = pd.DataFrame(columns=LUMI_CSV_COLUMNS)

        import luminoth.predict

        network = get_predictor_network(model, max_detections=100, min_prob=0.5)

        objects = luminoth.predict.predict_image(
            network, filename,
            only_classes=None,
            ignore_classes=None,
            save_path=None,
            min_prob=0.5,
            max_prob=1.0,
            pixel_distance=0,
            new_labels=None
        )
        for obj in objects:
            df = df.append({'image_id': filename,
                            'xmin': obj['bbox'][0],
                            'xmax': obj['bbox'][2],
                            'ymin': obj['bbox'][1],
                            'ymax': obj['bbox'][3],
                            'label': obj['label'],
                            'prob': obj["prob"]},
                           ignore_index=True)

        df = df.drop_duplicates()
        df.to_csv(csv_path, index=False)
        logger.info("lumi prediction per image subprocess call completed ")
        viewer.layers["Image"].metadata["lumi_inferenced"][index_of_image] = True
        inferenced_list[index_of_image] = True
        metadata = {"lumi_inferenced": inferenced_list}
        pickle_save(inference_metadata_path, metadata)
        load_bb_labels_for_image(viewer, csv_path)
    else:
        load_bb_labels_for_image(viewer, csv_path)
