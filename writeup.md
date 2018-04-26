**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/carExample.png
[notCar]: ./output_images/nonCarExample.png
[hogVisualcar]: ./output_images/HOG_visual_car.png
[hogVisualNotcar]: ./output_images/HOG_visual_not_car.png
[bboxes]: ./output_images/bboxes.png
[bboxes2]: ./output_images/bboxes2.png
[heatmap]: ./output_images/heatmap.png
[heatmap2]: ./output_images/heatmap2.png
[MeasurementLabel]: ./output_images/measurementLabel.png
[MeasurementLabel2]: ./output_images/measurementLabel2.png
[final]: ./output_images/final.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The services for this step are contained in the FeatureExcractionService class and implemented in the Pipeline.py lines 15 - 34.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car][car] ![Not a Car][notcar]

I shuffled the list of vehicles/non-vehicles before extracting the features.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  These parameter values can be found in the Params class. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![Hog Visual Car][hogVisualcar] ![Hog Visual Non Car][hogVisualNotcar]

#### 2. Explain how you settled on your final choice of HOG parameters.

After experimenting with various parameters I finally settled on the follow settings.

~~~~
class Params:
    train_new_svc = False
    test = False
    model_file_name = 'svc_model.save'
    scaler_filename = 'scaler.save'
    spatial_size = (32, 32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [375, 660]  # Min and max in y to search in slide_window()
~~~~

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn.svm LinearSVC library as well as sklearn.preprocessing for fitting a StandardScaler. The code can be found in the ClassifierTrainingService and implemented in the Pipeline.py line 36. After the classifier was trained I saved the svc and X_scaler to a pickle file in order to reuse them. There is a flag defined in the Params class that will train a new classifier if set to True.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use a sub-sampling window search that takes in an image, search area (for y axis), scale, trained classifier, x_scaler and other hog feature extraction params. This function can be found in HelperFunctions.py and called find_cars. It will extract features from an image, make a prediction for a specified window, draw a bounding box on the image if that prediction is correct, append the window to an array of correctly predicted windows and return the image along with the list of bounding boxes.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 1.5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Bounding Boxes][bboxes]
![Bounding Boxes2][bboxes2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 2 frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all both frames:
![Measurement label][MeasurementLabel]
![Measurement label2][MeasurementLabel2]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Final Result][final]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issue I had (and continue to have) is false positives and not having smooth transitions from frame to frame. I think a lot of improvements could be made the training of the classifier (reducing overfitting), the window search and the heatmap to reduce false positives. To make the project more robust I would gather more labeled data in order to train the a better classifier. I would also consider using a neural network to fine tune the training. I would also explore better ways to tune the window searching either by adjusting the parameters or breaking the find_cars method into smaller more tunable functions/services. Finally I didn't spend as much time as I would have liked to on removing false positives using heatmapping.

