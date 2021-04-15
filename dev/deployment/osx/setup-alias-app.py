###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
# 		   http://ilastik.org/license.html
###############################################################################
"""
This script uses py2app to generate an app bundle in "alias mode".
The resulting app is NOT suitable for redistribution, because it won't
contain any of the python modules needed for the app to run.
(They are all assumed to be available elsewhere on your system.)

However, if your environment is already "relocatable", then you can use the
"alias mode" app as template for a fully relocatable app.
"""

import sys
import os
from setuptools import setup, find_packages

import napari_bb_annotations

requirements = ["appdirs >= 1.4.4",
                "imageio-ffmpeg",
                "imageio>=2.5.0"
                "importlib-metadata>=1.5.0",
                "ipykernel>=5.1.1",
                "IPython>=7.7.0",
                "magicgui==0.2.8",
                "napari-plugin-engine>=0.1.5",
                "napari-svg>=0.1.3",
                "natsort",
                "numpy>=1.10",
                "pandas",
                "Pillow!=7.1.0,!=7.1.1,==7.2.0",  # not a direct dependency, but 7.1.0 and 7.1.1 broke imageio
                "psutil>=5.0",
                "PyOpenGL>=3.1.0",
                "pyside2>=5.12.3, <5.15",
                "PyYAML>=5.1",
                "qtconsole>=4.5.1",
                "qtpy>=1.7.0",
                "scipy>=1.2.0",
                "tensorflow==1.13.2",
                "toolz>=0.10.0",
                "vispy>=0.6.4",
                "wrapt>=1.11.1"]

if len(sys.argv) < 3 or sys.argv[1] != "py2app" or "--alias" not in sys.argv:
    sys.stderr.write("Usage: python {} py2app --alias ...\n".format(sys.argv[0]))
    sys.exit(1)

REPO = os.path.normpath(os.path.split(napari_bb_annotations.__file__)[0])
APP = ["bb_annotations.py"]

icon_file = os.path.join(os.path.split(__file__)[0], "icon.icns")
assert os.path.exists(icon_file)
OPTIONS = {
    "dist_dir": os.getcwd(),
    "site_packages": False,
    "argv_emulation": False,  # argv_emulation interferes with gui apps
    "iconfile": icon_file,
    "extra_scripts": [os.path.join(REPO, "mac_execfile.py")],
    "alias": True,
    "includes": "luminoth"
}

print(OPTIONS, REPO, APP)

setup(
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app==0.21"],
    install_requires=requirements,
    version="0.0.1",
    description="Bounding box annotations",
    packages=find_packages(exclude=["tests.*", "tests", "data"]),
    include_package_data=True,
    url="http://github.com/czbiohub/napari-bb-annotations",
)
