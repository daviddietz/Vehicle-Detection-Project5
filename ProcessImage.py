import numpy as np
import WindowService
from scipy.ndimage.measurements import label
from Params import Params
import HelperFunctions
from Model import Model
import matplotlib.pyplot as plt
import cv2


class Process(object):
    def process_image(self, image):
        test_draw_image = np.copy(image)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        windows = WindowService.slide_window(image, x_start_stop=[None, None], y_start_stop=Params.y_start_stop,
                                             xy_window=(96, 96), xy_overlap=(0.5, 0.5))

        hot_windows = WindowService.search_windows(image, windows, Model.svc, Model.X_scaler,
                                                   color_space=Params.color_space,
                                                   spatial_size=Params.spatial_size, hist_bins=Params.hist_bins,
                                                   orient=Params.orient, pix_per_cell=Params.pix_per_cell,
                                                   cell_per_block=Params.cell_per_block,
                                                   hog_channel=Params.hog_channel, spatial_feat=Params.spatial_feat,
                                                   hist_feat=Params.hist_feat, hog_feat=Params.hog_feat)

        # Add heat to each box in box list
        heat = HelperFunctions.add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = HelperFunctions.apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = HelperFunctions.draw_labeled_bboxes(np.copy(image), labels)
        if Params.test:
            window_img = HelperFunctions.draw_boxes(test_draw_image, hot_windows, color=(0, 0, 255), thick=6)
            plt.imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))
            # plt.show(block=True)

            print(labels[1], 'cars found')
            plt.imshow(labels[0], cmap='gray')

            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            plt.show(block=True)
            fig.tight_layout()
        return draw_img
