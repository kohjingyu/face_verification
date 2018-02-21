import sys
import dlib
from skimage import io
import time

import numpy as np

# Code modified from https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py

def get_images_for_username(username):
    ''' Returns the history of verified faces for username as a list of image paths '''
    # TODO: Implement retrieval of images from user folder
    history = ["images/Anthony/Anthony_Hopkins_0002.jpg", "images/Anthony/Anthony_Hopkins_0001.jpg", "images/Anthony/anthony-hopkins-6.jpg"]
    # image_path = "images/{0}/".format(username)

    return history

def verify_img(img, username):
    ''' Takes an img file and verifies it against user with username '''

    # Record time required
    ts = time.time()

    # Get history of verified faces
    gt_faces = get_images_for_username(username)

    # If no faces to compare against, return false (not verified)
    if len(gt_faces) == 0:
        return False

    # Get model paths
    predictor_path = "shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    # Load face detectors
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    face_rec = dlib.face_recognition_model_v1(face_rec_model_path)

    dets = detector(img, 1)
    total_distance = 0

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

            for gt_face in gt_faces:
                gt = io.imread(gt_face)
                gt_dets = detector(gt, 1)

                if len(gt_dets) != 1:
                    # There is more than one face or no faces
                    print("Error with ground truth. Please verify that it contains only one face.")
                else:
                    for k, d in enumerate(gt_dets):
                        # Compute the 128D vector for the history face
                        gt_shape = sp(gt, d)
                        gt_face_descriptor = face_rec.compute_face_descriptor(gt, gt_shape)
                        gt_v = np.asarray([x for x in gt_face_descriptor])

                        # Compare Euclidean distance
                        dist = np.linalg.norm(gt_v - v)
                        print("Euclidean distance: {0:.5f}".format(dist))
                        total_distance += dist

            # TODO: Output whether the faces are the same based on dist
            print("Average distance from all faces: {0:.5f}".format(total_distance / len(gt_faces)))

    # TODO: If this face is verified, add it to the list of verified faces, and compare against all of them in the future
    # TODO: Return true if verified, false otherwise

    # Output time taken
    print("Time elapsed: {0}s".format(time.time() - ts))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("To run: python face_recognition.py username face.jpg \n")
        exit()

    username = sys.argv[1] # the user we are verifying
    face_predict = sys.argv[2] # the image to verify

    print("Processing file: {}".format(face_predict))
    img = io.imread(face_predict)
    verified = verify_img(img, username)
