from qtpy.QtWidgets import QInputDialog, QLineEdit, QWidget, QFormLayout, QDialogButtonBox, QDialog
import os


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
        layout.addRow("Format of input", self.second)
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
        return path, format_of_files, box_annotations

    # save_gui_btn = QPushButton("save annotations [shift + s]")
    # save_gui_btn.clicked.connect(lambda: _save_shape(viewer))

    # viewer.window.add_dock_widget(
    #     [save_gui_btn],
    #     area="right",
    #     allowed_areas=["right", "left"],
    # )
