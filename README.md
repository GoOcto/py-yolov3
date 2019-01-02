# Python YoloV3

An android implementation of OpenCV-Yolov3 (compliments of darknet https://github.com/pjreddie/darknet)

## Installation

### Obtain neural net config files:

```
mkdir yolo-coco
wget https://pjreddie.com/media/files/yolov3.weights yolo-coco/yolov3.weights
wget https://github.com/pjreddie/darknet/tree/master/cfg/yolov3.cfg yolo-coco/yolov3.cfg
wget https://github.com/pjreddie/darknet/tree/master/data/coco.names yolo-coco/coco.names
```

### Install Python modules

```
pip install numpy
pip install opencv-python
```

### Process an image

Change the code in main.py to point to your image:

```
imagepath = "../yolo-object-detection/images/dining_table.jpg"
```

Run the code:

```
python main.py
```
