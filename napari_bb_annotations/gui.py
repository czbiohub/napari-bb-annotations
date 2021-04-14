from qtpy.QtWidgets import QPushButton
from ._key_bindings import (
    save_bb_labels,
    load_bb_labels,
    run_inference_on_images,
    run_lumi_on_image,
    run_segmentation_on_images,
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

    run_tflite_inference_btn = QPushButton("TensorflowLite Predict ALL images")
    run_tflite_inference_btn.clicked.connect(lambda: run_inference_on_images(viewer))
    run_tflite_inference_btn.setToolTip("Overwrites other predictions, Per 1 image with 20 bbs takes 40 ms/500 ms depending on Coral TPU connected/only CPU")

    run_segmentation_btn = QPushButton("Run thresholding to find bbs on ALL images")
    run_segmentation_btn.clicked.connect(lambda: run_segmentation_on_images(viewer))
    run_segmentation_btn.setToolTip("Overwrites other predictions, Runs otsu thresholding and converting to binary, and finds bbs")

    run_lumi_btn = QPushButton("Luminoth Predict on CURRENT image")
    run_lumi_btn.clicked.connect(lambda: run_lumi_on_image(viewer))
    run_lumi_btn.setToolTip("Overwrites other predictions, Per 1 image with 20 bbs takes 20 seconds depending on CPU/GPU")

    load_gui_btn = QPushButton("Load annotations")
    load_gui_btn.clicked.connect(lambda: load_bb_labels(viewer))
    load_gui_btn.setToolTip("Loads annotations from an existing file named bb_labels.csv created last time prediction was ran, not luminoth")

    save_gui_btn = QPushButton("Save annotations")
    save_gui_btn.clicked.connect(lambda: save_bb_labels(viewer))
    save_gui_btn.setToolTip("Saves annotations to bb_labels.csv inside your input images folder")

    edit_bb_label_btn = QPushButton("Edit label for bounding box")
    edit_bb_label_btn.clicked.connect(lambda: edit_bb_labels(viewer))
    edit_bb_label_btn.setToolTip("Edit label for 1 selected bounding box, use select tool that looks like a transparent arrow on left")

    viewer.window.add_dock_widget(
        [run_segmentation_btn,
         run_tflite_inference_btn,
         run_lumi_btn,
         load_gui_btn,
         save_gui_btn,
         edit_bb_label_btn],
        area="right",
        allowed_areas=["right", "left"],
    )
