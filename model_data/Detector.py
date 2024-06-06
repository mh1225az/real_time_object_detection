import cv2
import numpy as np
import time

np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening video file...")
            return

        (success, image) = cap.read()

        while success:
            classLabelIDs, confidences, bboxes = self.net.detect(image, confThreshold=0.5)

            bboxes = list(bboxes)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdxs = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdxs) != 0:
                for i in range(len(bboxIdxs)):
                    idx = np.squeeze(bboxIdxs[i])
                    bbox = bboxes[idx]
                    classConfidence = confidences[idx]
                    classLabelID = np.squeeze(classLabelIDs[idx])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                cv2.imshow("result", image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            success, image = cap.read()

        cap.release()
        cv2.destroyAllWindows()

