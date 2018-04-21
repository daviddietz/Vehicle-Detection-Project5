import glob
import cv2
import matplotlib.pyplot as plt
from FeatureExtractionService import FeatureExtractionService


class TestingPipline:
    spatial_size = (64, 64)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 0

    testCars = glob.glob('training_images/testCars/*.png')
    testNonCars = glob.glob('training_images/testNonCars/*.png')

    for car in testCars:
        image_car = cv2.imread(car)
        gray_car = cv2.cvtColor(image_car, cv2.COLOR_RGB2GRAY)
        carFeatures, hogImage = FeatureExtractionService().get_hog_features(gray_car, orient, pix_per_cell,
                                                                            cell_per_block,
                                                                            vis=True, feature_vec=False)

        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(gray_car, cmap='gray')
        plt.title('Car Image')
        plt.subplot(122)
        plt.imshow(hogImage, cmap='gray')
        plt.title('HOG Visualization')
        plt.show(block=True)

    for non_car in testNonCars:
        image_not_car = cv2.imread(non_car)
        gray_not_car = cv2.cvtColor(image_not_car, cv2.COLOR_RGB2GRAY)
        carNonFeatures, hogImage = FeatureExtractionService().get_hog_features(gray_not_car, orient, pix_per_cell,
                                                                               cell_per_block, vis=True,
                                                                               feature_vec=False)
        # Plot the examples
        figure = plt.figure()
        plt.subplot(121)
        plt.imshow(gray_not_car, cmap='gray')
        plt.title('None Car Image')
        plt.subplot(122)
        plt.imshow(hogImage, cmap='gray')
        plt.title('HOG Visualization')
        plt.show(block=True)

