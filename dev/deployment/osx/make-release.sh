set -e
set -x

RELEASE_VERSION=0.0.1

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
conda create -q -y -n napari-bb-annotations-release --file requirements.txt -c conda-forge python==3.7
conda activate napari-bb-annotations-release
${RELEASE_ENV}/bin/python setup.py install
conda install -y pyqt

${RELEASE_ENV}/bin/python dev/deployment/osx/setup-alias-app.py py2app --alias --dist-dir .

# proper QT-conf
cat <<EOF > he_annotations.app/Contents/Resources/qt.conf
; Qt Configuration file
[Paths]
Plugins = napari-bb-annotations-release/plugins
EOF

# add __main__ for convenience
cat <<EOF > he_annotations.app/Contents/Resources/he_annotations.py
from napari_he_annotations.he_annotations import main

if __name__ == "__main__":
    main()

EOF

# Moving napari-bb-annotations-release environment into he_annotations.app bundle
mv ${RELEASE_ENV} he_annotations.app/Contents/napari-bb-annotations-release

# Updating bundle internal paths
# Fix __boot__ script
sed -i '' 's|^_path_inject|#_path_inject|g' he_annotations.app/Contents/Resources/__boot__.py
sed -i '' "s|${CONDA_ROOT}/envs/napari-bb-annotations-release/||" he_annotations.app/Contents/Resources/__boot__.py
sed -i '' "s|DEFAULT_SCRIPT=.*|DEFAULT_SCRIPT='he_annotations.py'|" he_annotations.app/Contents/Resources/__boot__.py

# Fix Info.plist
sed -i '' "s|${CONDA_ROOT}/envs/napari-bb-annotations-release|@executable_path/../napari-bb-annotations-release|" he_annotations.app/Contents/Info.plist
sed -i '' "s|\.dylib|m\.dylib|" he_annotations.app/Contents/Info.plist

# Fix python executable link
rm he_annotations.app/Contents/MacOS/python
cd he_annotations.app/Contents/MacOS && ln -s ../napari-bb-annotations-release/bin/python
cd -

# Fix app icon link
rm he_annotations.app/Contents/Resources/icon.icns
cp dev/deployment/osx/icon.icns he_annotations.app/Contents/Resources

# Replace Resources/lib with a symlink
rm -rf he_annotations.app/Contents/Resources/lib
cd he_annotations.app/Contents/Resources && ln -s ../napari-bb-annotations-release/lib
cd -

# Rename app bundle
mv he_annotations.app napari-bb-annotations_${RELEASE_VERSION}.app

# tar app bundle
tar -cjf napari-bb-annotations_${RELEASE_VERSION}.app.tar.bz2 napari-bb-annotations_${RELEASE_VERSION}.app/

