from qtpy.QtWidgets import QInputDialog, QLineEdit, QWidget, QFormLayout, QDialogButtonBox, QDialog, QPushButton
from magicgui.widgets import ComboBox, Container
import os
import numpy as np


def create_label_menu(viewer, labels):
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
    label_menu = ComboBox(label='text label', choices=labels)
    label_widget = Container(widgets=[label_menu])
    return label_widget


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        self.third = QLineEdit(self)
        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Images/Video to annotate with bounding boxes", self.first)
        layout.addRow("Format of input, including .", self.second)
        layout.addRow("Comma separated classes you want to annotate in the images", self.third)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text(), self.third.text())


def connect_to_viewer(viewer):
    """
    Add gui elements to the viewer
    """
    dialog = InputDialog()
    if dialog.exec():
        path, format_of_files, box_annotations = dialog.getInputs()
        box_annotations = box_annotations.split(",")
        assert os.path.exists(os.path.abspath(path)), "Path provided {} doesn't exist, please restart".format(path)
        label_widget = create_label_menu(viewer, box_annotations)
        # add the label selection gui to the viewer as a dock widget
        viewer.window.add_dock_widget(label_widget, area='right')
        return path, format_of_files, box_annotations

    # save_gui_btn = QPushButton("save annotations [shift + s]")
    # save_gui_btn.clicked.connect(lambda: _save_shape(viewer))

    # viewer.window.add_dock_widget(
    #     [save_gui_btn],
    #     area="right",
    #     allowed_areas=["right", "left"],
    # )
