BOX_ANNOTATIONS=classone classtwo classthree
PATH=./data/jpg_images/
FORMAT_OF_FILES=.jpg

napari-viewer:
	python napari_bb_annotations/bb_annotations.py \
		--path ${PATH} \
		--format_of_files ${FORMAT_OF_FILES} \
		--box_annotations ${BOX_ANNOTATIONS} \

napari-viewer-from-app:
	napari-bb-annotations_0.0.1.app/Contents/MacOS/bb_annotations \
		--path ${PATH} \
		--format_of_files ${FORMAT_OF_FILES} \
		--box_annotations ${BOX_ANNOTATIONS} \