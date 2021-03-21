import datetime
import logging
import os
import re
import numpy as np
import pandas as pd

from qtpy.QtWidgets import (QInputDialog, QLineEdit, QWidget, QFormLayout,
                            QDialogButtonBox, QDialog, QPushButton)
from magicgui.widgets import ComboBox, Container
from PIL import ImageDraw
from napari import Viewer

from ._key_bindings import update_layers, save_bb_labels, load_bb_labels, run_inference_on_images


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        self.third = QLineEdit(self)
        self.fourth = QLineEdit(self)
        self.fifth = QLineEdit(self)

        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow(
            "Images/Video to annotate with bounding boxes, labels", self.first)
        layout.addRow("Format of input, including .", self.second)
        layout.addRow(
            "Comma separated classes you want to annotate in the images",
            self.third)
        layout.addRow(
            "Tensorflow lite model to use to inference",
            self.fourth)
        layout.addRow(
            "Enter true to use coral TPU connected to your computer while inferencing",
            self.fifth)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text(), self.third.text(), self.fourth.text(), self.fifth.text())


def connect_to_viewer(viewer, path, format_of_files, box_annotations, model, edgetpu):
    """
    Add gui elements to the viewer
    """
    if (path and format_of_files and box_annotations) is not None:
        dialog = InputDialog()
        if dialog.exec():
            path, format_of_files, box_annotations, model, edgetpu = dialog.getInputs()
    update_gui_btn = QPushButton(
        "Update layers, only click once per stack")
    update_gui_btn.clicked.connect(
        lambda: update_layers(viewer, box_annotations))

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
    return path, format_of_files, box_annotations, model, edgetpu
