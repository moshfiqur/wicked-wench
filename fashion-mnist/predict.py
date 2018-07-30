import tensorflow as tf
import os
import cv2
import numpy as np
import glob

ROOT_DIR = os.getcwd()
DATA_DIR = ROOT_DIR + '/data'
RESOURCE_DIR = ROOT_DIR + '/images'
MODEL_DIR = ROOT_DIR + '/model'

# class info
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
            'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sess = tf.Session()
# sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('model/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = tf.get_default_graph()

x = graph.get_tensor_by_name('x:0')
y = graph.get_tensor_by_name('y:0')

img_size = 28

path = os.path.join('images', 'test', '*g')
files = glob.glob(path)
for img_file in files:
    print('working on ', img_file)
    test_im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    # test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
    # test_im = cv2.resize(test_im, (img_size, img_size))
    
    # invert grayscale
    test_im = 255 - cv2.resize(test_im, (img_size, img_size), cv2.INTER_LINEAR)
    
    # plt.imshow(test_im, cmap=plt.get_cmap('Greys_r'))
    # plt.show()
    # test_im = test_im.reshape(1, img_size, img_size, num_channels)
    test_im = test_im.flatten().reshape(1, 784)
    # print(test_im.shape)
    # cv2.imshow('image', test_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(test_im.shape)

    # THIS GETS OUR LABEL AS A INTEGER
    # label = classes[y_train.argmax()]
    # label = 'dummy'
    
    # THIS GETS OUR PREDICATION AS A INTEGER
    prediction = sess.run(y, feed_dict={x: test_im})
    # print(prediction)
    prediction = prediction.argmax()

    print(classes[prediction])
    
    # plt.title('Prediction: %d Label: %s' % (prediction, classes[prediction]))
    # plt.imshow(test_im.reshape([28,28]), cmap=plt.get_cmap('Greys_r'))
    # plt.show()
