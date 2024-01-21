import glob

from FaceRecognitionCLS import *


threshold = 70  # percentage

if __name__ == "__main__":

    model = FaceRecognitionClass()
    # faces to be tested for detection, recognition and proper bounding
    model.set(threshold)
    model.encodefaces()  # encodes faces in the faces folder as  our database

    # test user faces of jpg files in TEST folder
    # the results are written in TEST_RESULTS folder
    for filename in glob.glob(os.path.join(".//TEST", '*.jpg')):
        if os.path.isfile(filename):
            model.recognizeface(filename)


