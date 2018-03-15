import unittest
from face_exceptions import MultiFaceException, NoFaceException, NoHistoryException
import face_recognition

class TestFaceVerification(unittest.TestCase):
    def test_no_faces(self):
        # Test with an image of an elephant
        with self.assertRaises(NoFaceException):
            face_recognition.verify_img("images/test_cases/elephant.jpg", "anthony_hopkins")

    def test_multi_face(self):
        # Test with an image of multiple people
        with self.assertRaises(MultiFaceException):
            face_recognition.verify_img("images/test_cases/party.jpg", "anthony_hopkins")

    def test_no_history(self):
        # Test user with no previously uploaded images
        with self.assertRaises(NoHistoryException):
            face_recognition.verify_img("images/test_cases/tom_hiddleston.jpg", "tom_hiddleston")

    def test_different_face(self):
        # Test with a different person
        verified = face_recognition.verify_img("images/test_cases/sudipta-chattopadhyay.jpg", "anthony_hopkins")
        self.assertFalse(verified)

    def test_verified_face(self):
        # Test with a verified face (of Anthony Hopkins)
        verified = face_recognition.verify_img("images/test_cases/anthony-hopkins.jpg", "anthony_hopkins")
        self.assertTrue(verified)

if __name__ == "__main__":
    unittest.main()