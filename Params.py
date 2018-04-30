class Params:
    train_new_svc = False
    test = False
    model_file_name = 'svc_model.save'
    scaler_filename = 'scaler.save'
    spatial_size = (32, 32)
    hist_bins = 64
    orient = 10
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    n_frames = 10
