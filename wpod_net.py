import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "models/wpod-net.json"
wpod_net = load_model(wpod_net_path)
print(wpod_net.summary())
# def preprocess_image(image_path,resize=False):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img / 255
#     if resize:
#         img = cv2.resize(img, (224,224))
#     return img


# forward image through model and return plate's image and coordinates
# if error "No Licensese plate is founded!" pop up, try to adjust Dmin
# def get_plate(image_path, Dmax=608, Dmin=256):
#     vehicle = preprocess_image(image_path)
#     ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
#     side = int(ratio * Dmin)
#     bound_dim = min(side, Dmax)
#     _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
#     return LpImg, cor

# def get_plate_from_car(image):
#     cv2.imwrite("wpod_temp.jpg", image)
#     LpImg, cor = get_plate("wpod_temp.jpg")

#     print(f"============ HELLO NUMBER PLATES WE GOT IS {len(LpImg)} ============")
#     images = []
#     for el in LpImg:
#         cv2.imwrite("wpod_temp.jpg", np.round(el*255))
#         image = cv2.imread("wpod_temp.jpg")
#         images.append(image[:, :, ::-1])
#     return images
