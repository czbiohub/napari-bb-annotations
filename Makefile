BOX_ANNOTATIONS=class1 class2 class3 class4
PATH=./data/
FORMAT_OF_FILES=.jpeg

napari-viewer:
	python napari_bb_annotations/launcher/bb_annotations.py \
		--path ${PATH} \
		--format_of_files ${FORMAT_OF_FILES} \
		--box_annotations ${BOX_ANNOTATIONS} \

napari-viewer-from-app:
	napari-bb-annotations_0.0.1.app/Contents/MacOS/bb_annotations \
		--path ${PATH} \
		--format_of_files ${FORMAT_OF_FILES} \
		--box_annotations ${BOX_ANNOTATIONS} \