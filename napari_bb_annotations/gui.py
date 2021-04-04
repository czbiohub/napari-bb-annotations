from qtpy.QtWidgets import QPushButton, QLabel
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

    # make a tooltop about the prediction
    tooltip = QLabel()
    qtext = "<p>Prediction on 1 image takes 20 seconds</p>"
    tooltip.setText(qtext)

    run_inference_btn = QPushButton("Tensorflow Predict ALL images, click first if never ran")
    run_inference_btn.clicked.connect(lambda: run_inference_on_images(viewer))

    run_inference_single_image_btn = QPushButton("Tensorflow Predict ONE CURRENT image")
    run_inference_single_image_btn.clicked.connect(lambda: run_inference_on_images(viewer))

    load_gui_btn = QPushButton("Load annotations click if tensorflow prediction was run before")
    load_gui_btn.clicked.connect(lambda: load_bb_labels(viewer))

    save_gui_btn = QPushButton("Save annotations")
    save_gui_btn.clicked.connect(lambda: save_bb_labels(viewer))

    edit_bb_label_btn = QPushButton("Edit label for bounding box")
    edit_bb_label_btn.clicked.connect(lambda: edit_bb_labels(viewer))

    viewer.window.add_dock_widget(
        [tooltip],
        area="right",
        allowed_areas=["right", "left"],
    )

    viewer.window.add_dock_widget(
        [run_inference_btn,
         run_inference_single_image_btn,
         load_gui_btn,
         save_gui_btn, edit_bb_label_btn],
        area="right",
        allowed_areas=["right", "left"],
    )
