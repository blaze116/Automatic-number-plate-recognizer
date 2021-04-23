
					
						#Automatic license plate recognition in unconstrained conditions


**File description**

| **FILENAME / FOLDER NAME**               | **CONTENTS**                                                 |
| ---------------------------------------- | ------------------------------------------------------------ |
| **classes.names and darknet-yolov3.cfg** | configuration files for YOLO model.                          |
| **obj_det.py**                           | contains code to run inference on an image using trained YOLO Model |
| **local_utils.py and wpod_net.py**       | contains code to run inference on an image using trained WPOD-NET. |
| **ptransform.py**                        | contains code to run the four-point perspective transformation on an image. |
| **seg_rec.py**                           | contains code to run segmentation and recognition on an image using the trained Detectron 2 model. |

**Approach**

Here is a brief explanation of our approach. Our approach can be divided into three parts :

1) License Plate detection

2) Perspective transformation of LP

3) Segmentation and recognition of characters
**1) License Plate detection** **:**	

For plate detection, we used Yolo v3, a variant of Darknet, which originally is a 53 layer network trained on Imagenet. Yolo performed so well on multiple car images but it was missing number plates of distant cars. So along with this we included another object detection model WPOD-NET, in order to increase the robustness of our solution. Also because of its smaller size as compared to yolo it didn’t have much effect on efficiency of license plate detection tasks.

**2) Perspective transformation of LP** **:**

After the first step, we are left with an image of LP which might contain a tilted or warped license plate. So we now perform the four-point perspective transformation on the image, an example of it can be seen below.

<img src="assets\ptr.png" alt="ptr" style="zoom:80%;" />

**3) Segmentation and recognition** **:**We use Detectron 2, Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. We used Mask R-CNN R50-FPN variant which is used for instance segmentation and classification in a robust manner.

**Results**

<img src="assets\res1.png" alt="res1" style="zoom:80%;" />



**Model Training**

Detectron 2 is widely used because of its speed in training and inference, the classification loss, and the bounding-box regression loss converged just after training for a few minutes. We chose Mask R-CNN R50-FPN variant and trained it using transfer learning with initial weights trained on COCO dataset.

**Installation**

<img src="assets\12.png" alt="12" style="zoom: 67%;" />

​				

**Usage** 

Here is an example to run inference on images that includes just plates : 

<img src="assets\2.png" alt="2" style="zoom: 67%;" />

Here is an example to run inference on images that includes cars (can also include multiple cars) : 

<img src="assets\3.png" alt="3" style="zoom:67%;" />

Here is an example to run inference on a video : 

<img src="assets\4.png" alt="4" style="zoom:67%;" />

**Inference Time**

Time taken to run inference on Nvidia Tesla K80 GPU is 2 sec per image.Time taken to run inference on CPU is 5 sec per Image.

**Team**

Payal Umesh Pote

Vishwas Chepuri

Ananya Singh

Weights link - https://drive.google.com/file/d/1ap9npAmNW7B5kuHbgJrPDz4LWNzfkvEC/view?usp=sharing
