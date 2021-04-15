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

Deploy the app using `./dev/deployment/osx/make-release.sh` if a .app package was not provided to you, otherwise on 
MacOS, download the .app file and double click on it

If you were given a .bz2 file, double-click to extract the .app file from the archive, double click again


Once the app opens up, 
1. Drag and drop the images folder
2. Click `Tensorflow Predict ALL images` to predict. Currently the app is set to predict on this branch using a luminoth [checkpoint](https://github.com/czbiohub/luminoth-uv-imaging/releases) these 6 classes "Culex erythrothorax","Culex tarsalis", "Culiseta inornata", "Debris", "Insect", "Mosquito" . You do not have to click `load annotations` again to load
3. Click `Tensorflow Predict ONLY CURRENT image` to predict the current image only as each image can take upto 10-20 seconds to predict. If you predict a whole folder of 100 images, be patient and wait upto 30 minutes. You do not have to click `load annotations` again to load
3. Click the `Load annotations` button if you have a bb_labels.csv annotations that was saved previously you can reload it. 
4. Note to correct both bounding box and the label, use select on top left  to select the bounding box or bounding boxes you want to change the label of and change the label in drop down menu on bottom right
5. If you want to edit or add a new bounding box use `Edit label for bounding box` button after `Select shapes` and update with the new name on the bounding box pop up on the left and close it afterwards using the `x` that says `hide the panel`
5. Click the `Select shapes` button to delete the selected box and label using `x` symbol on the left among different arrows for the shape layer
7. Click `Save annotations`, classes along with the image path to `bb_labels.csv` or bounding boxes, labels overlaid image, you can click `s` any number of times as you progress, the overlay directory is created inside the path with the name `overlay_dir`
8. Write frame number near the frame slider to skip to a slide, otherwise clicking the slider right and left arrows will make you switch before different frames
9. The input if prediction is already ran, csv output is a file named `bb_labels.csv` file contains xmin, xmax, ymin, ymax, image_id, label. `Overlaid_dir` contains bounding boxes and labels overlaid on original images in the input folder. The `inference_metadata.pickle` file keeps track of your annotations, progress, and enables loading. The `summary_data.json` file contains the summary counts, percentages per class. 

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
