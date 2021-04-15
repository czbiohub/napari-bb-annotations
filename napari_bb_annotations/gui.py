from qtpy.QtWidgets import QPushButton
from ._key_bindings import (
    save_bb_labels,
    load_bb_labels,
    run_inference_on_images,
    edit_bb_labels)


def connect_to_viewer(viewer):
    """
    Add gui elements to the viewer
    """
    # Turn off ipython console if it is on
    view = viewer.window.qt_viewer
    # Check if console is present before it is requested
    if view._console is not None:
        # Check console has been created when it is supposed to be shown
        view.toggle_console_visibility(None)

    run_inference_btn = QPushButton("Tensorflow Predict ALL images")
    run_inference_btn.clicked.connect(lambda: run_inference_on_images(viewer))
    run_inference_btn.setToolTip("Prediction per 100 images with 20 bbs takes upto 30 minutes depending on CPU/GPU")

    run_inference_single_image_btn = QPushButton("Tensorflow Predict ONLY CURRENT image")
    run_inference_single_image_btn.clicked.connect(lambda: run_inference_on_images(viewer))
    run_inference_single_image_btn.setToolTip("Prediction per image with 20 bbs takes upto 20 seconds depending on CPU/GPU")

    load_gui_btn = QPushButton("Load annotations")
    load_gui_btn.clicked.connect(lambda: load_bb_labels(viewer))
    load_gui_btn.setToolTip("Loads annotations from an existing file named bb_labels.csv created last time prediction was ran")

    save_gui_btn = QPushButton("Save annotations")
    save_gui_btn.clicked.connect(lambda: save_bb_labels(viewer))
    save_gui_btn.setToolTip("Saves annotations to bb_labels.csv inside your input images folder")

    edit_bb_label_btn = QPushButton("Edit label for bounding box")
    edit_bb_label_btn.clicked.connect(lambda: edit_bb_labels(viewer))
    edit_bb_label_btn.setToolTip("Edit label for 1 selected bounding box, use select tool that looks like a transparent arrow on left")

    viewer.window.add_dock_widget(
        [run_inference_btn,
         run_inference_single_image_btn,
         load_gui_btn,
         save_gui_btn, edit_bb_label_btn],
        area="right",
        allowed_areas=["right", "left"],
    )
