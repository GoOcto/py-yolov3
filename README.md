# Python YoloV3

An android implementation of OpenCV-Yolov3 (compliments of darknet https://github.com/pjreddie/darknet)

## Installation

### Obtain neural net weight file:

```
wget https://pjreddie.com/media/files/yolov3.weights yolo-coco/yolov3.weights
```

### Install Python modules

```
pip install numpy
pip install opencv-python
```

### Process an image

Change the code in main.py to point to your image:

```
imagepath = "images/dining_table.jpg"
```

Run the code:

```
python main.py
```

### Or process video

Change the code in video.py to point to your video:

```
src = "videos/sample.mp4"
```

Run the code:

```
python video.py
```
