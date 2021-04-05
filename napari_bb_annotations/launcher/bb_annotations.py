import napari

from napari_bb_annotations.gui import connect_to_viewer


def launch_viewer():
    with napari.gui_qt():
        viewer = napari.Viewer()
        connect_to_viewer(viewer)


def main():
    launch_viewer()


if __name__ == "__main__":
    main()
