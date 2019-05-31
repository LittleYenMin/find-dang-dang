# from mrcnn.model import MaskRCNN
import os
import mrcnn.utils
import mrcnn.model
import mrcnn.config


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Download COCO trained weights from Releases if needed
if not os.path.exists(os.getenv('COCO_MODEL_PATH')):
    mrcnn.utils.download_trained_weights(os.getenv('COCO_MODEL_PATH'))

# Create a Mask-RCNN model in inference mode
model = mrcnn.model.MaskRCNN(mode="inference", model_dir=os.getenv('R_CNN_MODEL_DIR'), config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(os.getenv('COCO_MODEL_PATH'), by_name=True)
