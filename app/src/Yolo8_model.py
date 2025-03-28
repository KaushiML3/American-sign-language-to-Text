from ultralytics import YOLO
import cv2
import os

current_path =os.path.dirname(os.path.abspath(__file__))
# Load a pretrained YOLOv8n model
model = YOLO(os.path.join(current_path,"model/yolo8/weights/best.pt"))



# Open the video file
#video_path = "predict\Learn ASL Alphabet Video.mp4"


# Read the image
#image_path = 'predict\istockphoto-1222186261-612x612_jpg.rf.80d02f491fba51b640c166610f32eaf5.jpg'
#image = cv2.imread(image_path)


def yolo_model(image):

# Run inference
    results = model.predict(source=image,save=False,stream=True,show=True, conf=0.25)

    # Process and print results
    for result in results:
        bboxes = result.boxes.xyxy
        labels = result.boxes.cls
        scores = result.boxes.conf

        for bbox, label, score in zip(bboxes, labels, scores):
            print(f"Label: {label}, Score: {score}, BBox: {bbox}")
            return(f"Label: {label}, Score: {score}, BBox: {bbox}")



# Run inference
results = model.predict(source=0,save=False,stream=True,show=True, conf=0.50)

# Process and print results
for result in results:
    bboxes = result.boxes.xyxy
    labels = result.boxes.cls
    scores = result.boxes.conf

    for bbox, label, score in zip(bboxes, labels, scores):
        print(f"Label: {label}, Score: {score}, BBox: {bbox}")
       
