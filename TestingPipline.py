import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ClassifierTrainingService
import FeatureExtractionService
import WindowService
import HelperFunctions
from scipy.ndimage.measurements import label


class TestingPipline:
    spatial_size = (32, 32)
    hist_bins = 16
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [400, 660]  # Min and max in y to search in slide_window()
    x_start_stop = [100, 1200]

    testCars = glob.glob('training_images/testCars/*.png')
    testNonCars = glob.glob('training_images/testNonCars/*.png')

    car_features = FeatureExtractionService.extract_features(testCars, color_space=color_space,
                                                               spatial_size=spatial_size,
                                                               hist_bins=hist_bins, orient=orient,
                                                               pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                               hog_channel=hog_channel,
                                                               spatial_feat=True, hist_feat=True, hog_feat=True)

    non_car_features = FeatureExtractionService.extract_features(testNonCars, color_space=color_space,
                                                                   spatial_size=spatial_size,
                                                                   hist_bins=hist_bins, orient=orient,
                                                                   pix_per_cell=pix_per_cell,
                                                                   cell_per_block=cell_per_block,
                                                                   hog_channel=hog_channel,
                                                                   spatial_feat=True, hist_feat=True, hog_feat=True)

    svc, X_scaler = ClassifierTrainingService.train_linear_svm(car_features=car_features, non_car_features=non_car_features)

    image = cv2.imread('test_images/test3.jpg')
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    draw_image = np.copy(image)

    windows = WindowService.slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = WindowService.search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = HelperFunctions.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))
    plt.show(block=True)

    # ystart = 400
    # ystop = 656
    # scale = 4
    #
    # out_img = HelperFunctions.find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)
    #
    # plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    # plt.show(block=True)

    # Add heat to each box in box list
    heat = HelperFunctions.add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = HelperFunctions.apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = HelperFunctions.draw_labeled_bboxes(np.copy(image), labels)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.show(block=True)
    fig.tight_layout()
