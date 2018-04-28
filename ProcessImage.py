import numpy as np
import WindowService
import FindCarService
from scipy.ndimage.measurements import label
from Params import Params
import HelperFunctions
from Model import Model
import matplotlib.pyplot as plt
from collections import deque


class Process(object):
    heatmaps = deque(maxlen=Params.n_frames)

    def process_image(self, image):
        draw_image = np.copy(image)
        # image = image.astype(np.float32) / 255

        # windows = WindowService.slide_window(image, x_start_stop=[None, None], y_start_stop=Params.y_start_stop,
        #                                      xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        #
        # hot_windows = WindowService.search_windows(image, windows, Model.svc, Model.X_scaler,
        #                                            color_space=Params.color_space,
        #                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins,
        #                                            orient=Params.orient, pix_per_cell=Params.pix_per_cell,
        #                                            cell_per_block=Params.cell_per_block,
        #                                            hog_channel=Params.hog_channel, spatial_feat=Params.spatial_feat,
        #                                            hist_feat=Params.hist_feat, hog_feat=Params.hog_feat)

        bboxes = []

        bboxes.append(FindCarService.find_cars(image, ystart=380,
                                            ystop=450, scale=1.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=380,
                                            ystop=465, scale=1.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=380,
                                            ystop=480, scale=1.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=380,
                                            ystop=480, scale=1.5, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=400,
                                            ystop=500, scale=1.5, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=425,
                                            ystop=525, scale=1.5, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=380,
                                            ystop=525, scale=2.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=410,
                                            ystop=525, scale=2.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=430,
                                            ystop=560, scale=2.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=425,
                                            ystop=550, scale=3.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=450,
                                            ystop=675, scale=3.0, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=380,
                                            ystop=590, scale=3.5, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        bboxes.append(FindCarService.find_cars(image, ystart=430,
                                            ystop=675, scale=3.5, svc=Model.svc,
                                            X_scaler=Model.X_scaler, orient=Params.orient,
                                            pix_per_cell=Params.pix_per_cell, cell_per_block=Params.cell_per_block,
                                            spatial_size=Params.spatial_size, hist_bins=Params.hist_bins))

        #Flatten bbox list
        #Thanks to Alex Martelli via stack overflow
        #https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
        bboxes = [item for sublist in bboxes for item in sublist]

        heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heatmap = HelperFunctions.add_heat(heatmap, bboxes)

        self.heatmaps.append(heatmap)
        combined = sum(self.heatmaps)

        # Apply threshold to help remove false positives
        heatmap = HelperFunctions.apply_threshold(combined, 8)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heatmap, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        result_image = HelperFunctions.draw_labeled_bboxes(draw_image, labels)

        if Params.test:
            window_img = HelperFunctions.draw_boxes(draw_image, bboxes, color=(0, 0, 255), thick=6)

            plt.imshow(window_img)
            plt.show(block=True)

            # print(labels[1], 'cars found')
            # plt.imshow(labels[0], cmap='gray')

            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(result_image)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            plt.show(block=True)
            fig.tight_layout()

        return result_image
