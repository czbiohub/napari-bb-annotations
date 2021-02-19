from PyQt5.QtWidgets import (QFormLayout, QApplication,QWidget,QPushButton,QLineEdit,QInputDialog,QHBoxLayout, QDialog, QDialogButtonBox)
import sys


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self);

        layout = QFormLayout(self)
        layout.addRow("First text", self.first)
        layout.addRow("Second text", self.second)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text())

if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    dialog = InputDialog()
    if dialog.exec():
        print(dialog.getInputs())
    exit(0)
