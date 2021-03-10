from qtpy.QtWidgets import QInputDialog, QLineEdit, QWidget, QFormLayout, QDialogButtonBox, QDialog, QPushButton
from magicgui.widgets import ComboBox, Container
import os
import numpy as np


# create the GUI for selecting the values
def create_label_menu(shapes_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters:
    -----------
    shapes_layer : napari.layers.Shapes
        a napari shapes layer
    label_property : str
        the name of the shapes property to use the displayed text
    labels : List[str]
        list of the possible text labels values.

    Returns:
    --------
    label_widget : magicgui.widgets.Container
        the container widget with the label combobox
    """
    # Create the label selection menu
    label_property = text_property = "box_label"
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
        """This is acallback that update the current properties on the Shapes layer
        when the label menu selection changes
        """
        selected_label = event.value
        current_properties = shapes_layer.current_properties
        current_properties[label_property] = np.asarray([selected_label])
        shapes_layer.current_properties = current_properties

    label_menu.changed.connect(label_changed)

    return label_widget


def update_layers(viewer, labels):
    label_widget = create_label_menu(viewer, box_annotations)
    label_property = text_property = "box_label"
    text_color = 'black'
    size = 10
    shapes_layer = viewer.layers['shape']

    # To fix bug in currently have for creating emtpy layers with text
    # see: https://github.com/napari/napari/issues/2115
    def on_data(event):
        if viewer.layers['shape'].text.mode == 'none':
            viewer.layers['shape'].text = text_property
            viewer.layers['shape'].text.color = text_color
            viewer.layers['shape'].text.size = size
    viewer.layers['shape'].events.set_data.connect(on_data)
    # add the label selection gui to the viewer as a dock widget
    viewer.window.add_dock_widget(label_widget, area='right')


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        self.third = QLineEdit(self)
        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow(
            "Images/Video to annotate with bounding boxes", self.first)
        layout.addRow("Format of input, including .", self.second)
        layout.addRow(
            "Comma separated classes you want to annotate in the images",
            self.third)

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
        assert os.path.exists(
            os.path.abspath(path)),
        "Path provided {} doesn't exist, please restart".format(path)
        update_gui_btn = QPushButton("Update layers, only click once [u]")
        update_gui_btn.clicked.connect(
            lambda: update_layers(viewer, box_annotations))

        viewer.window.add_dock_widget(
            [update_gui_btn],
            area="right",
            allowed_areas=["right", "left"],
        )
        return path, format_of_files, box_annotations
