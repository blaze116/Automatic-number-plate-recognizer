import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import matplotlib.pyplot as plt
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

mapping = {
    "0" : 0, "1" : 1, "2" : 2, "3" : 3, "4" : 4, "5" : 5,
    "6" : 6, "7" : 7, "8" : 8, "9" : 9, "A" : 10, "B" : 11, 
    "C" : 12, "D" : 13, "E" : 14, "F" : 15, "G" : 16, 
    "H" : 17, "I" : 18, "J" : 19, "K" : 20, "L" : 21,
    "M" : 22, "N" : 23, "O" : 24, "P" : 25, "Q" : 26,
    "R" : 27, "S" : 28, "T" : 29, "U" : 30, "V" : 31,
    "W" : 32, "X" : 33, "Y" : 34, "Z" : 35
}

reverse_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 
                    6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 
                    12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 
                    18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 
                    24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 
                    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

def get_character_dicts(img_dir):
    dataset_dicts = []
    for idx, filename in enumerate(os.listdir(img_dir)):
        if filename[-4:] != "json":
            continue
        with open(f"{img_dir}/{filename}") as f:
            img_anns = json.load(f)
        record = {}
        record["file_name"] = "data/" + filename[:-4] + "png"
        record["image_id"] = idx
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]
        objs = []
        for anno in img_anns["shapes"]:
            x1 = round(anno["points"][0][0])
            y1 = round(anno["points"][0][1])
            x2 = round(anno["points"][1][0])
            y2 = round(anno["points"][1][1])
            temp = anno["label"].capitalize()

            obj = {
                "bbox" : [x1, y1, x2, y2],
                "bbox_mode" : BoxMode.XYXY_ABS,
                "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
                "category_id" : mapping[temp]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train"]:
    DatasetCatalog.register("char_" + d, lambda d=d: get_character_dicts("data"))
    MetadataCatalog.get("char_" + d).set(thing_classes=[*mapping.keys()])
balloon_metadata = MetadataCatalog.get("data")

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("char_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0008   
cfg.SOLVER.MAX_ITER = 3500    
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 36 

# Required for training
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()

cfg.MODEL.WEIGHTS ="models/model_final_with_cls.pth" # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def get_preds(outputs, image):
    temp = outputs["instances"].pred_boxes.tensor.to('cpu').numpy()
    val_preds = list(outputs["instances"].pred_classes.to('cpu').numpy())
    scores = list(outputs["instances"].scores.to('cpu').numpy())

    preds = []
    for el1, el2, el3 in zip(temp, val_preds, scores):
        preds.append([el1, el2, el3])
    preds.sort(key=lambda x:x[0][0])

    iou_thresh = 0.75
    prev = preds[0]
    crct_preds = []
    for el in preds[1:]:
        if bb_intersection_over_union(prev[0], el[0]) < iou_thresh:
            crct_preds.append(prev)
            prev = el
        else:
            if prev[-1] < el[-1]:
                prev = el
    crct_preds.append(prev)
    ans = ""
    for el in crct_preds:
        ans += reverse_mapping[el[-2]]
    plot_chars(crct_preds, image)
    return ans

def plot_chars(preds, image):
    fig = plt.figure()
    for i, el in enumerate(preds):
        plt.subplot(1, len(preds), i+1)
        x1, y1, x2, y2 = el[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        plt.imshow(image[y1:y2, x1:x2, :])
    plt.show()


from detectron2.utils.visualizer import ColorMode

def get_chars(im):
    
    im = cv2.resize(im, (2000*im.shape[1]//im.shape[0], 2000))
    output = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # print(output)
    # print(f"For image {d} -> ", get_preds(output))
    return get_preds(output, im)


