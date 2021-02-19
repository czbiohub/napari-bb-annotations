import napari
import os
import imageio
from magicgui.widgets import ComboBox, Container
import numpy as np
import logging
import skimage.color
from skimage.io import imread, ImageCollection
import datetime
from gui import connect_to_viewer

# set up the annotation values and text display properties
# create the GUI for selecting the values
text_property = 'box_label'
text_color = 'green'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create file handler which logs even info messages
now = datetime.datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
fh = logging.FileHandler("bb_annotations_{}.log".format(date_time), "w")
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
formatter = logging.Formatter(
    "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

IMAGE_FORMATS = [".png", ".tiff", ".tif", ".jpg", ".jpeg"]


def create_label_menu(shapes_layer, label_property, labels):
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


def imread_convert(f):
    return imread(f).astype(np.uint8)


def launch_viewer():
    with napari.gui_qt():
        viewer = napari.Viewer()
        path, format_of_files, box_annotations = connect_to_viewer(viewer)
        if format_of_files not in IMAGE_FORMATS:
            image_collection = imageio.get_reader(path, 'ffmpeg')
        else:
            image_collection = ImageCollection(
                path + os.sep + "*" + format_of_files,
                load_func=imread_convert)

        for image in image_collection:
            viewer = napari.view_image(image)
            shapes = viewer.add_shapes(
                properties={text_property: box_annotations})
            shapes.text = 'box_label'

            # create the label section gui
            label_widget = create_label_menu(
                shapes_layer=shapes,
                label_property=text_property,
                labels=box_annotations
            )
            # add the label selection gui to the viewer as a dock widget
            viewer.window.add_dock_widget(label_widget, area='right')

            # set the shapes layer mode to adding rectangles
            shapes.mode = 'add_rectangle'

        # this is a hack to get around a bug we currently have for creating emtpy layers with text
        # see: https://github.com/napari/napari/issues/2115
        def on_data(event):
            if shapes.text.mode == 'none':
                shapes.text = text_property
                shapes.text.color = text_color
        shapes.events.set_data.connect(on_data)


def main():
    launch_viewer()


if __name__ == "__main__":
    main()
