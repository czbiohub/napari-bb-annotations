# napari-bb-annotations

[![License](https://img.shields.io/pypi/l/napari-bb-annotations.svg?color=green)](https://github.com/pranathivemuri/napari-bb-annotations/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-bb-annotations.svg?color=green)](https://pypi.org/project/napari-bb-annotations)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-bb-annotations.svg?color=green)](https://python.org)
[![tests](https://github.com/pranathivemuri/napari-bb-annotations/workflows/tests/badge.svg)](https://github.com/pranathivemuri/napari-bb-annotations/actions)
[![codecov](https://codecov.io/gh/pranathivemuri/napari-bb-annotations/branch/master/graph/badge.svg)](https://codecov.io/gh/pranathivemuri/napari-bb-annotations)

Bounding box annotations

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `napari-bb-annotations` via [pip]:

    git clone https://github.com/czbiohub/napari-bb-annotations.git
    cd napari-bb-annotations
    conda env create -n napari-bb-annotations -f environment.yml
    conda activate napari-bb-annotations
    python setup.py install
    python napari_bb_annotations/launcher/bb_annotations.py \
        --path ./data/jpgstoannotated/ \
		--format_of_files .jpg \
		--box_annotations class1 class2 class3 `

Or

MacOS: download the .app file or deploy the app file using `./dev/deployment/osx/make-release.sh` and then right click on the .app file created and get the path to app as below and provide the arguments to your path, format_of_files, box_annotations

`/Users/pranathi.vemuri/Downloads/napari-he-annotations_0.0.1.app/Contents/MacOS/bb_annotations --path ./data/jpgstoannotated/ --format_of_files .jpg --box_annotations class1 class2 class3 `

If you were given a .bz2 file, double-click to extract the .app file from the archive. Right-click the .app file and get the path to the app by clicking on `Get info` and provide the arguments to your path, format_of_files, box_annotations, as above


Once the app opens up, 
1. Enter the path of your folder containing `images` or `video` subfolders into the file dialog like so using flags `--path /Users/pranathi.vemuri/napari-bb-annotations/data/`, `--format_of_files .jpg` and `box_annotations class1 class2 class3 class4 `
2. If you are opening the app using cli and have already given paths, please click the cancel button when it asks for path, box_annotations, and format_of_files again
3. Click the `Shift-l` key or load annotations button if you have a csv annotations that was saved previously you can reload it. Ignore this step if you do not have the annotations previously saved as well as press this only once per stack. Once you click load the bounding boxes are updated but not the label yet, Step 6 actually enables the correct labels, i.e click `Update layers` after load if you have previous annotations, otherwise just press `Update layers`
4. Note to correct both bounding box and the label, use select on top left  to select the bounding box or bounding boxes you want to change the label of and change the label in drop down menu on bottom right
5. Click the `Select shapes` button to delete the selected box and label using `x` symbol on the left among different arrows for the shape layer
6. Click `Update layers` button only once per stack to see the dock with the `box_annotations` 
7. Click `Shift-s` to save the bounding boxes, classes along with the image path to `bb_labels.csv` or bounding boxes, labels overlaid image, you can click `s` any number of times as you progress, the overlay directory is created inside the path with the name `overlay_dir`
8. Write frame number near the frame slider to skip to a slide, otherwise clicking the slider right and left arrows will make you switch before different frames
9. To run inference, give the inputs of model i.e the path to tflite file, edgetpu flag (set it if you have a coral connected) - The model will use tflite_runtime to infer/predict the bounding boxes, labels when you click `run inference ` button and to load those annotations click load annotations. Running inference should be the first steps if you want to figure out bounding boxes, labels and correct them

To use inferencing, install tflite_runtime for mac with 3.5 as below, to figure out a different wheel based on your platform go here - https://github.com/google-coral/pycoral/releases/

`pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl`

To install edgetpu dependencies alongside the tflite_runtime

`curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt-get install edgetpu-compiler`



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-bb-annotations" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/pranathivemuri/napari-bb-annotations/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
