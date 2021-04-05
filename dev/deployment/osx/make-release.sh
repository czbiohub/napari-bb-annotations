set -e
set -x

RELEASE_VERSION=0.0.3

echo "Creating app bundle for napari-bb-annotations ${RELEASE_VERSION}"

# run from napari-bb-annotations folder

# setup conda
eval "$(conda shell.bash hook)"
conda activate base
CONDA_ROOT=`conda info --root`
source ${CONDA_ROOT}/bin/activate root

RELEASE_ENV=${CONDA_ROOT}/envs/napari-bb-annotations-release

# remove old release environment
conda env remove -y -q -n napari-bb-annotations-release

echo "Creating new release environment"
conda env create -n napari-bb-annotations-release -f environment.yml
conda activate napari-bb-annotations-release
${RELEASE_ENV}/bin/python setup.py install
conda install -y pyqt
pip install git+git://github.com/czbiohub/luminoth-uv-imaging.git@6a5115c395c915a4f8390732182e5c32d0c30794 # master - Tested for only until April 4th commit of master
pip install git+git://github.com/napari/napari.git@866848d35d039c098b45e8d6f12eae0924633347 # master - Tested for only until April 4th commit of master
lumi predict --help

${RELEASE_ENV}/bin/python dev/deployment/osx/setup-alias-app.py py2app --alias --dist-dir .  --packages=wx --emulate-shell-environment

# proper QT-conf
cat <<EOF > bb_annotations.app/Contents/Resources/qt.conf
; Qt Configuration file
[Paths]
Plugins = napari-bb-annotations-release/plugins
EOF

# add __main__ for convenience
cat <<EOF > bb_annotations.app/Contents/Resources/bb_annotations.py
from napari_bb_annotations.launcher.bb_annotations import main

if __name__ == "__main__":
    main()

EOF

# Moving napari-bb-annotations-release environment into bb_annotations.app bundle
mv ${RELEASE_ENV} bb_annotations.app/Contents/napari-bb-annotations-release

# Updating bundle internal paths
# Fix __boot__ script
sed -i '' 's|^_path_inject|#_path_inject|g' bb_annotations.app/Contents/Resources/__boot__.py
sed -i '' "s|${CONDA_ROOT}/envs/napari-bb-annotations-release/||" bb_annotations.app/Contents/Resources/__boot__.py
sed -i '' "s|DEFAULT_SCRIPT=.*|DEFAULT_SCRIPT='bb_annotations.py'|" bb_annotations.app/Contents/Resources/__boot__.py

# Fix Info.plist
sed -i '' "s|${CONDA_ROOT}/envs/napari-bb-annotations-release|@executable_path/../napari-bb-annotations-release|" bb_annotations.app/Contents/Info.plist
sed -i '' "s|\.dylib|m\.dylib|" bb_annotations.app/Contents/Info.plist

# Fix python executable link
rm bb_annotations.app/Contents/MacOS/python
cd bb_annotations.app/Contents/MacOS && ln -s ../napari-bb-annotations-release/bin/python
cd -

# Fix app icon link
rm bb_annotations.app/Contents/Resources/icon.icns
cp dev/deployment/osx/icon.icns bb_annotations.app/Contents/Resources

# Replace Resources/lib with a symlink
rm -rf bb_annotations.app/Contents/Resources/lib
cd bb_annotations.app/Contents/Resources && ln -s ../napari-bb-annotations-release/lib
cd -

# Rename app bundle
mv bb_annotations.app napari-bb-annotations_${RELEASE_VERSION}.app

# tar app bundle
tar -cjf napari-bb-annotations_${RELEASE_VERSION}.app.tar.bz2 napari-bb-annotations_${RELEASE_VERSION}.app/

