from qtpy.QtWidgets import QPushButton
from ._key_bindings import (
    save_bb_labels,
    load_bb_labels,
    run_inference_on_image,
    edit_bb_labels,
    update_layers)


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

    update_gui_btn = QPushButton(
        "Update layers, only click once per stack")
    update_gui_btn.clicked.connect(
        lambda: update_layers(viewer))

    save_gui_btn = QPushButton("Save annotations")
    save_gui_btn.clicked.connect(lambda: save_bb_labels(viewer))

    load_gui_btn = QPushButton("Load annotations if existing, summary table")
    load_gui_btn.clicked.connect(lambda: load_bb_labels(viewer))

    run_inference_btn = QPushButton("Predict bounding box, labels in 1 image using tensorflow")
    run_inference_btn.clicked.connect(lambda: run_inference_on_image(viewer))

    edit_bb_label_btn = QPushButton("Edit label for bounding box")
    edit_bb_label_btn.clicked.connect(lambda: edit_bb_labels(viewer))

    viewer.window.add_dock_widget(
        [update_gui_btn, save_gui_btn, load_gui_btn,
         run_inference_btn, edit_bb_label_btn],
        area="right",
        allowed_areas=["right", "left"],
    )
