from lib.lib import *

class data():
    def dataTrain(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        # reshape images to specify that it's a single channel
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

        def preprocess_images(imgs): # should work for both a single image and multiple images
            sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
            assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape # make sure images are 28x28 and single-channel (grayscale)
            return imgs / 255.0

        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)
        
        return train_images, train_labels, test_images, test_labels