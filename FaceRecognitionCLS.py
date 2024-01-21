import math
import os
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw


class FaceRecognitionClass:

    def __init__(self):
        self.__known_label = []
        self.__known_encoded_faces = []
        self.__threshold = 60  # percentage

    def set(self, threshold):
        self.__threshold = threshold

    def getlabel(self, face_file_name):

        label = ""
        for each_char in range(len(face_file_name)):
            if not ("0" <= str(face_file_name[each_char]) <= "9"):
                cur_char = face_file_name[each_char]
                label = label + cur_char
            else:
                break

        return label

    def faceconfidence(self, face_distance, face_match_threshold=0.6):
        range = (1 - face_match_threshold)
        linear_v = (1 - face_distance) / (range * 2)

        if face_distance > face_match_threshold:
            return round(linear_v * 100, 2)
        else:
            val = (linear_v + ((1 - linear_v) * math.pow((linear_v - 0.5) * 2, 0.5))) * 100
            return round(val, 2)

    def encodefaces(self):

        for each_face in os.listdir('./faces'):
            img = face_recognition.load_image_file(f'faces/{each_face}')

            face_encoding = face_recognition.face_encodings(img)[0]
            self.__known_encoded_faces.append(face_encoding)
            label = self.getlabel(each_face)
            self.__known_label.append(label)

    def recognizeface(self, path_test):

        if os.path.isfile(path_test):

            test = face_recognition.load_image_file(path_test)
            test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

            # detect and encode faces in current frame
            face_locations = face_recognition.face_locations(test)
            face_encodings = face_recognition.face_encodings(test, face_locations)

            # Loop through faces in test image
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.__known_encoded_faces, face_encoding)
                face_distances = face_recognition.face_distance(self.__known_encoded_faces, face_encoding)

                name = "unknown"
                confidence = None
                found = False
                bboxes = None

                if True in matches:
                    # a face is detecting matching with a person in our database
                    best_match_ind = np.argmin(face_distances)
                    confidence = self.faceconfidence(face_distances[best_match_ind])
                    if confidence > self.__threshold:
                        # with high confidence
                        name = self.__known_label[best_match_ind]
                        found = True
                        bboxes = (top, right, bottom, left)

                        cv2.rectangle(test, (left, top), (right, bottom), color=(255, 255, 255))
                        cv2.putText(test, name + " " + str(confidence) + " %", (left + 6, bottom - 5),
                                    color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)

                    else:
                        # with low confidence
                        name = self.__known_label[best_match_ind]
                        found = True
                        cv2.rectangle(test, (left, top), (right, bottom), color=(255, 255, 255))
                        cv2.putText(test,
                                    name + " " + str(confidence) + "%", (left + 6, bottom - 5),
                                    color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
                        cv2.putText(test,
                                    "below threshold " + str(self.__threshold) + "%",
                                    (left + 6, bottom + 25),
                                    color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)

                else:
                    # a face is detected, but it is not in our database
                    cv2.rectangle(test, (left, top), (right, bottom), color=(255, 255, 255))
                    cv2.putText(test, name, (left + 6, bottom - 5),
                                color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)

            cv2.imwrite('./TEST_RESULTS/' + path_test[7:-4] + ".jpg", test)
