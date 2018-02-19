import sys
import dlib
from skimage import io
import time

import numpy as np

# Code modified from https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py

if len(sys.argv) != 3:
    print("To run: python face_recognition.py gt_face.jpg face2.jpg\n")
    exit()

predictor_path = "shape_predictor_5_face_landmarks.dat" #sys.argv[1]
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat" #sys.argv[2]
gt_face = sys.argv[1] # ground truth - the image we are verifying against
face_predict = sys.argv[2] # the image to verify

# Record time required
ts = time.time()

# Load face detectors
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
face_rec = dlib.face_recognition_model_v1(face_rec_model_path)

print("Processing file: {}".format(face_predict))
img = io.imread(face_predict)
dets = detector(img, 1)

if(len(dets) != 1):
    print("Error: More than one face!")
else:
    # Now process each face we found.
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. Here we just print
        # the vector to the screen.
        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        v = np.asarray([x for x in face_descriptor])

        # Face 1 is the groundtruth: we compare face 2 to face 1 to verify face 2
        gt = io.imread(gt_face)
        gt_dets = detector(gt, 1)

        if len(gt_dets) != 1:
        	# There is more than one face or no faces
        	print("Error with ground truth. Please verify that it contains only one face.")
        else:
	        for k, d in enumerate(gt_dets):
	            gt_shape = sp(gt, d)
	            gt_face_descriptor = face_rec.compute_face_descriptor(gt, gt_shape)
	            gt_v = np.asarray([x for x in gt_face_descriptor])

	            dist = np.linalg.norm(gt_v - v)
	            print("Euclidean distance: {0:.5f}".format(dist))

	            # TODO: Output whether the faces are the same based on dist

# TODO: If this face is verified, add it to the list of verified faces, and compare against all of them in the future
print("Time elapsed: {0}s".format(time.time() - ts))