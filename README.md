# face_verification

Performs face comparison between two specified images and outputs their Euclidean distance. This is used for our 50.003 application, a Know-Your-Customer verification app. For more details, please visit the [main repository](https://github.com/132lilinwei/newsite).

## Requirements

Our image processing functionality is based off the wonderful toolkit, [dlib](https://github.com/davisking/dlib).

We use several libraries for face verification and image processing. To install this, several dependencies are required:

```
pip install numpy
pip install scipy
pip install scikit-image
pip install requests
pip install Pillow
pip install cmake
pip install dlib

```

*Note:* some of these libraries are rather large, and will take a while to install.


## Unit Tests

To run unit tests for the image verification module, execute

```
python unit_test.py
```

within a terminal window in the root directory of the repository.

### Test Cases

We check for several edge cases and common image issues:

* Image with no faces throws an exception
* Image with more than one face throws an exception
* Testing a user with no history of images uploaded throws an exception
* Testing a user with an incorrect photo throws an exception
* Testing a verified photo passes


