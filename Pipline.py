import glob


class Pipeline:
    # cars = glob.glob('training_images/vehicles/*.png')
    # notcars = glob.glob('training_images/non_vehicles/*.png')
    testCars = glob.glob('training_images/testCars/*.png')
    testNonCars = glob.glob('training_images/testNonCars/*.png')
