import napari

from napari_bb_annotations.gui import connect_to_viewer


if __name__ == "__main__":
    with napari.gui_qt():
        viewer = napari.Viewer()
        connect_to_viewer(viewer)
