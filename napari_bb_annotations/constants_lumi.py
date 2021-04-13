EDGETPU = False
BOX_ANNOTATIONS = [
    "healthy",
    "ring",
    "schizont",
    "troph"
]
MODEL = "1a0f3002f674"
IMAGE_FORMATS = [".png", ".tiff", ".tif", ".jpg", ".jpeg"]
LUMI_CSV_COLUMNS = [
    'image_id', 'xmin', 'xmax', 'ymin', 'ymax',
    'label', 'prob', "unique_cell_id"]
EDGE_COLOR_CYCLE = ['green', 'magenta', 'blue', 'red']
TFLITE_MODEL = "https://github.com/czbiohub/napari-bb-annotations/raw/pranathi-ulc/data/output_tflite_graph.tflite"
