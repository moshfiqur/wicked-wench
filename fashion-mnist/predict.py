import tensorflow as tf
import os
import cv2
import numpy as np
import glob

ROOT_DIR = os.getcwd()
DATA_DIR = ROOT_DIR + '/data'
RESOURCE_DIR = ROOT_DIR + '/images'
MODEL_DIR = ROOT_DIR + '/model'

# All classes for categorization
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
            'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sess = tf.Session()

# Load and restore the trained model
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = tf.get_default_graph()

# Get the tensors which we will be needed
# for the prediction. The names must be 
# assigned during the training.
x = graph.get_tensor_by_name('x:0')
y = graph.get_tensor_by_name('y:0')

img_size = 28

path = os.path.join('images', 'test', '*g')
files = glob.glob(path)
for img_file in files:
    print('working on ', img_file)
    
    # Load the image in grayscale
    test_im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    
    # Or convert an already loaded color image to grayscale
    # test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
    
    # Resize an image
    # test_im = cv2.resize(test_im, (img_size, img_size))
    
    # resize image and invert grayscale
    test_im = 255 - cv2.resize(test_im, (img_size, img_size), cv2.INTER_LINEAR)
    
    # If needed, see the loaded, resized and inverted image
    # from jupyter notebook
    # plt.imshow(test_im, cmap=plt.get_cmap('Greys_r'))
    # plt.show()
    
    # Reshape the image when using color images
    # test_im = test_im.reshape(1, img_size, img_size, num_channels)
    
    # Flatten the image and then reshape it to convert the 
    # matrix image (28x28) into a vector of size (1, 784)
    test_im = test_im.flatten().reshape(1, 784)
    
    # If needed, see the image (from python script)
    # cv2.imshow('image', test_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # THIS GETS OUR LABEL AS A INTEGER
    # label = classes[y_train.argmax()]
    
    # Lets get the prediction as integer which will be the 
    # key in the classes list above
    prediction = sess.run(y, feed_dict={x: test_im}).argmax()

    print(classes[prediction])
    
    # When needed, check the prediction output with image
    # plt.title('Prediction: %d Label: %s' % (prediction, classes[prediction]))
    # plt.imshow(test_im.reshape([28,28]), cmap=plt.get_cmap('Greys_r'))
    # plt.show()
