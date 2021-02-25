import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from keras.layers import Input

PROJECT_PATH = str(str(os.path.realpath(__file__).replace('\\', '/')).split('frcnn-pod/')[0]) + 'frcnn-pod/'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from train import nn_base, rpn_layer, classifier_layer
from keras.models import Model


if __name__ == '__main__':

    base_test_path = 'frcnn-pod/test/'

    test_img_path = 'test_images'
    test_xml_path = 'test_xml'

    config_output_filename = 'model/model_vgg_config.pickle'

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    C.record_path = 'model/record.csv'
    C.model_path = 'model/model_frcnn_vgg.hdf5'

    # Load the records
    record_df = pd.read_csv(C.record_path)

    r_epochs = len(record_df)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')
    plt.savefig(base_test_path + 'mean_overlapping_bboxes_class_acc.png')
    plt.close()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.savefig(base_test_path + 'loss_rpn_cls_loss_rpn_regr.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.savefig(base_test_path + 'loss_class_cls_loss_class_regr.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
    plt.title('elapsed_time')
    plt.savefig(base_test_path + 'total_loss_elapsed_time.png')
    plt.close()

    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
