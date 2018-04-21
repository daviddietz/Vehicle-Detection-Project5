import glob
from FeatureExtractionService import FeatureExtractionService


class Pipeline:
    spatial_size = (32, 32)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 0
    color_space = 'RGB'
    cars = glob.glob('training_images/vehicles/*.png')
    not_cars = glob.glob('training_images/non_vehicles/*.png')

    carFeatures = FeatureExtractionService().extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                                              hist_bins=hist_bins, orient=orient,
                                                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                              hog_channel=hog_channel,
                                                              spatial_feat=True, hist_feat=True, hog_feat=True)
    nonCarFeatures = FeatureExtractionService().extract_features(not_cars, color_space=color_space,
                                                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                                                 orient=orient, pix_per_cell=pix_per_cell,
                                                                 cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                                 spatial_feat=True, hist_feat=True, hog_feat=True)
