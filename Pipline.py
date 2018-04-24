from ProcessImage import Process
import glob
import pickle
from sklearn.externals import joblib
import FeatureExtractionService
import ClassifierTrainingService
from Params import Params
import numpy as np
import HelperFunctions

import cv2
import WindowService
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


class Pipeline:
    if Params.train_new_svc:
        cars = glob.glob('training_images/vehicles/*.png')
        not_cars = glob.glob('training_images/non_vehicles/*.png')

        car_features = FeatureExtractionService.extract_features(cars, color_space=Params.color_space,
                                                                 spatial_size=Params.spatial_size,
                                                                 hist_bins=Params.hist_bins, orient=Params.orient,
                                                                 pix_per_cell=Params.pix_per_cell,
                                                                 cell_per_block=Params.cell_per_block,
                                                                 hog_channel=Params.hog_channel,
                                                                 spatial_feat=True, hist_feat=True, hog_feat=True)
        non_car_features = FeatureExtractionService.extract_features(not_cars, color_space=Params.color_space,
                                                                     spatial_size=Params.spatial_size, hist_bins=Params.hist_bins,
                                                                     orient=Params.orient, pix_per_cell=Params.pix_per_cell,
                                                                     cell_per_block=Params.cell_per_block,
                                                                     hog_channel=Params.hog_channel,
                                                                     spatial_feat=True, hist_feat=True, hog_feat=True)

        svc, X_scaler = ClassifierTrainingService.train_linear_svm(car_features=car_features,
                                                                   non_car_features=non_car_features)
        pickle.dump(svc, open(Params.model_file_name, 'wb'))
        joblib.dump(X_scaler, Params.scaler_filename)

    image = cv2.imread('test_images/test6.jpg')
    Process().process_image(image)

    # project_video = 'output_video.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # test_clip = clip1.fl_image(Process().process_image)
    # test_clip.write_videofile(project_video, audio=False)
