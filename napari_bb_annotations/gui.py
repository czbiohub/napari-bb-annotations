from qtpy.QtWidgets import QPushButton
from ._key_bindings import (
    update_layers,
    save_bb_labels,
    load_bb_labels,
    run_inference_on_images)
from napari_bb_annotations.constants_lumi import BOX_ANNOTATIONS


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
        lambda: update_layers(viewer, BOX_ANNOTATIONS))

    save_gui_btn = QPushButton("save annotations [shift + s]")
    save_gui_btn.clicked.connect(lambda: save_bb_labels(viewer))

    load_gui_btn = QPushButton("load annotations [shift + l]")
    load_gui_btn.clicked.connect(lambda: load_bb_labels(viewer))

    run_inference_btn = QPushButton("Run inference [shift + i]")
    run_inference_btn.clicked.connect(lambda: run_inference_on_images(viewer))

    viewer.window.add_dock_widget(
        [update_gui_btn, save_gui_btn, load_gui_btn, run_inference_btn],
        area="right",
        allowed_areas=["right", "left"],
    )
