import cv2
import torch
import time
import speech_recognition as sr
import os

# Camera Measurements
# Macbook Pro 2019 13':
# 1) 0,81-0,85 for 62±1 cm
# 2) 0,28-0,32 for 30±1 cm

# SYNTAX
# "<object> and <object> and <object>..."

objectList = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

# objectList = ['cat', 'dog', 'backpack', 'umbrella', 'handbag', 'bottle', 'wineglass', 'cup', 'fork', 'knife',
#               'spoon', 'bowl', 'chair', 'couch', 'potted plant', 'dining table', 'tv', 'laptop', 'mouse',
#               'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors']

def get_input():
        print("What are you looking for?")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
                audio = recognizer.listen(source)
                said = ""

                try:
                        # CAN BE ADJUSTED
                        said = recognizer.recognize_google(audio)
                        # said=said[5:]
                        thingsToFind = said.split(" and ")
                        return thingsToFind
                except Exception as e:
                        print("Exception: " + str(e))


def depthToDist(depth):
    distance = 60 * (1 - depth) + 12.2
    distance = round(distance, 2)
    result = "Depth: " + str(distance) + " cm"
    return result

# start
inputList = get_input()

print("You said {}".format(inputList))

objectsToFind = []

for object in inputList:
        if object in objectList:
                objectsToFind.append(object.lower())
        else:
                print("I don't know what " + str(object) + " is.")
print("Looking for {}".format(objectsToFind))

recColor = (0, 255, 0)
recThickness = 3
capture = cv2.VideoCapture(1)

start_time = time.time()

if capture.isOpened():
    print("Capture open")
else:
    raise IOError("Capture not working")

# Model: YOLOv5s / Custom
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# yolo = torch.hub.load('ultralytics/yolov5', 'custom',
#                       path='/Users/alexandrubaila/Desktop/GOFORE/video_streaming/best-2.pt',
#                       force_reload=True)

# Load GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# get MiDaS depth map
path_model = "models/"
model_name = "model-small.onnx";  # MiDaS v2.1 Small

midas = cv2.dnn.readNet(path_model + model_name)

end_time=0
fps=0

while True:
    _, frame = capture.read()

    height, width, _ = frame.shape

    # FPS computing
    end_time=time.time()
    fps = round(1/(end_time-start_time), 2)
    start_time = end_time
    frameWithFPS = cv2.putText(frame, str(fps) + " FPS", (17, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    # get YOLO detections
    outputYOLO = yolo(frame).pandas().xyxy[0]

    # compute Depth Map with MiDaS
    blob = cv2.dnn.blobFromImage(frame, 1 / 255., (256, 256), (123.675, 116.28, 103.53), True, False)

    midas.setInput(blob)
    outputMIDAS = midas.forward()

    outputMIDAS = outputMIDAS[0, :, :]
    outputMIDAS = cv2.resize(outputMIDAS, (width, height))
    depthMap = cv2.normalize(outputMIDAS, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # output Depth Map
    cv2.imshow("MiDaS", depthMap)

    for index in range(outputYOLO.shape[0]):
        row = outputYOLO.iloc[index]
        classification = row['name']

        if classification in objectsToFind:

            topLeft = (int(row.xmin.item()), int(row.ymin.item()))
            bottomRight = (int(row.xmax.item()), int(row.ymax.item()))

            # detection center point, for depth computation
            centerPoint = (int(row.xmax.item() / 2), int(row.ymax.item() / 2))

            inverseDepth = depthMap[centerPoint] # 0,78 - 0,83 for 82-87 cm real distance
            realDepth = depthToDist(inverseDepth)

            confidence = str(round(row.confidence.item(), 2))
            info = classification + " (" + confidence + ")"
            infoCoord = (int(row.xmin.item()), int(row.ymin.item()) - 5)
            depthCoord = (int(row.xmin.item()), int(row.ymin.item()) + 35)

            frameWithFPS = cv2.rectangle(frameWithFPS, topLeft, bottomRight, recColor, recThickness)
            frameWithFPS = cv2.putText(frameWithFPS, info, infoCoord, cv2.FONT_HERSHEY_SIMPLEX, 1, recColor, 2, cv2.LINE_AA)
            frameWithFPS = cv2.putText(frameWithFPS, realDepth, depthCoord, cv2.FONT_HERSHEY_SIMPLEX, 1, recColor, 2, cv2.LINE_AA)


    # output camera
    cv2.imshow("Camera", frameWithFPS)

    if cv2.waitKey(1) == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()

