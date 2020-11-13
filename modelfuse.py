
import os
import tensorflow as tf
import numpy as np
from ops import *

def Network(input, training):

    side_layer1, side_layer2, side_layer3, side_layer4, side_layer5 = Basenet(input,training)
    seg1_3_up, seg3_3_up, seg5_3_up, segfuse, seg5_3= Segnet(side_layer1, side_layer3, side_layer5, training)
    stem1_3_up, stem2_3_up, stem3_3_up, stemfuse = Stemnet(side_layer1, side_layer2, side_layer3, seg5_3, training)
    
    return seg1_3_up, seg3_3_up, seg5_3_up, segfuse, stem1_3_up, stem2_3_up, stem3_3_up, stemfuse


def Basenet(input,training, reuse=False):
    #print(input)
    #print(training)
    net = conv2d_block(x=input, c=16, k=5, s=1, is_train=training, BR=True, name='sidelayer1')
    sidelayer1=net
    print('sidelayer1',sidelayer1)
    with tf.variable_scope("main_network", reuse=reuse):
        residual_list = get_residual_layer(10)
        exp = 6
        ########################################################################################################
        net = res_block(net, exp, 16, 2, is_train=training, name='resblock1_0')
        for i in range(1,residual_list[0]):
            net = res_block(net, exp, 16, 1, is_train=training, name='resblock1_' + str(i))
        sidelayer2 = net
        print('sidelayer2',sidelayer2)
        ########################################################################################################
        net = res_block(net, exp, 32, 2, is_train=training, name='resblock2_0')
        for i in range(1,residual_list[1]):
            net = res_block(net, exp, 32, 1, is_train=training, name='resblock2_' + str(i))
        sidelayer3 = net
        print('sidelayer3',sidelayer3)
        ########################################################################################################
        net = res_block(net, exp, 32, 2, is_train=training, name='resblock3_0')
        for i in range(1, residual_list[2]):
            net = res_block(net, exp, 32, 1, is_train=training, name='resblock3_'+ str(i))
        sidelayer4 = net
        print('sidelayer4',sidelayer4)
        ########################################################################################################
        net = res_block(net, exp, 32, 2, is_train=training, name='resblock4_0')
        for i in range(1, residual_list[3]):
            net = res_block(net, exp, 32, 1, is_train=training, name='resblock4_' + str(i))
        sidelayer5 = net
        print('sidelayer5',sidelayer5)
        ########################################################################################################
        return sidelayer1,sidelayer2,sidelayer3,sidelayer4,sidelayer5

def Segnet(seg1,seg3,seg5,training):

    seg1_1 = conv2d_block(x=seg1, c=16, k=3, s=1, is_train=training, BR=True, name='seg1_1')
    seg3_1 = conv2d_block(x=seg3, c=16, k=3, s=1, is_train=training, BR=True, name='seg3_1')
    seg5_1 = conv2d_block(x=seg5, c=16, k=3, s=1, is_train=training, BR=True, name='seg5_1')

   ## seg1_1_to_1_2 = conv2d_block(x=seg1_1, c=16, k=3, s=1, is_train=training, BR=True, name='seg1_1_to_1_2')
    seg3_1_to_1_2 = deconv2d(seg3_1, f_shape=[8, 8, 16, 16], output_shape=[5, 300, 400, 16], stride=4, name='seg3_1_to_1_2')
  #  seg1_2_concat = tf.concat([seg3_1_to_1_2, seg1_1_to_1_2], axis=-1, name='seg1_2_concat')
    seg1_2_concat = tf.concat([seg3_1_to_1_2, seg1_1], axis=-1, name='seg1_2_concat')
    seg1_2 = conv2d_block(x=seg1_2_concat, c=16, k=3, s=1, is_train=training, BR=True, name='seg1_2')
    seg1_3 = conv2d_block(x=seg1_2, c=1, k=3, s=1, is_train=training, BR=False, name='seg1_3')

    seg1_1_to_3_2 = conv2d_block(x=seg1_1, c=16, k=1, s=4, is_train=training, BR=True, name='seg1_1_to_3_2')
   ## seg3_1_to_3_2 = conv2d_block(x=seg3_1, c=16, k=3, s=1, is_train=training, BR=True, name='seg3_1_to_3_2')
    seg5_1_to_3_2 = deconv2d(seg5_1, f_shape=[8, 8, 16, 16], output_shape=[5, 75, 100, 16], stride=4, name='seg5_1_to_3_2')
    # seg3_2_concat = tf.concat([seg1_1_to_3_2, seg3_1_to_3_2, seg5_1_to_3_2], axis=-1, name='seg3_2_concat')
    seg3_2_concat = tf.concat([seg1_1_to_3_2, seg3_1, seg5_1_to_3_2], axis=-1, name='seg3_2_concat')
    seg3_2 = conv2d_block(x=seg3_2_concat, c=16, k=3, s=1, is_train=training, BR=True, name='seg3_2')
    seg3_3 = conv2d_block(x=seg3_2, c=1, k=3, s=1, is_train=training, BR=False, name='seg3_3')

  ##  seg5_1_to_5_2 = conv2d_block(x=seg5_1, c=16, k=3, s=1, is_train=training, BR=True, name='seg5_1_to_5_2')
    seg3_1_to_5_2 = conv2d_block(x=seg3_1, c=16, k=1, s=4, is_train=training, BR=True, name='seg3_1_to_5_2')
    # seg5_2_concat = tf.concat([seg5_1_to_5_2, seg3_1_to_5_2], axis=-1, name='seg5_2_concat')
    seg5_2_concat = tf.concat([seg5_1, seg3_1_to_5_2], axis=-1, name='seg5_2_concat')
    seg5_2 = conv2d_block(x=seg5_2_concat, c=16, k=3, s=1, is_train=training, BR=True, name='seg5_2')
    seg5_3 = conv2d_block(x=seg5_2, c=1, k=3, s=1, is_train=training, BR=False, name='seg5_3')
    
    # seg1_2_up = deconv2d(seg1_2, f_shape=[4, 4, 1, 1], output_shape=[5, 300, 400, 1], stride=2, name='seg1_2_up')
    # seg3_2_up = deconv2d(seg3_2, f_shape=[16, 16, 1, 1], output_shape=[5, 300, 400, 1], stride=8, name='seg3_2_up')
    # seg5_2_up = deconv2d(seg5_2, f_shape=[64, 64, 1, 1], output_shape=[5, 300, 400, 1], stride=32, name='seg5_2_up')
    seg1_3_up = seg1_3
    seg3_3_up = deconv2d(seg3_3, f_shape=[8, 8, 1, 1], output_shape=[5, 300, 400, 1], stride=4, name='seg3_3_up')
    seg5_3_up = deconv2d(seg5_3, f_shape=[32, 32, 1, 1], output_shape=[5, 300, 400, 1], stride=16, name='seg5_3_up')
   
   
    # segfuse_concat = tf.concat([seg1_2_up, seg3_2_up, seg5_2_up], axis=-1, name='segfuse_concat')
    segfuse_concat = tf.concat([seg1_3_up, seg3_3_up, seg5_3_up], axis=-1, name='segfuse_concat')
    segfuse =  tf.layers.conv2d(segfuse_concat, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                    name='segfuse', kernel_initializer=tf.constant_initializer(
                                    0.33)) 
    
    return seg1_3_up, seg3_3_up, seg5_3_up, segfuse, seg5_3


def Stemnet(stem1, stem2, stem3, seg5_3, training):
    
    seg5_3 = tf.nn.sigmoid(seg5_3)

    
    
    

    stem5_2_to_stem3_1 = deconv2d(seg5_3, f_shape=[8, 8, 1, 1], output_shape=[5, 75, 100, 1], stride=4, name='stem5_2_to_stem3_1')
    stem5_2_to_stem2_1 = deconv2d(seg5_3, f_shape=[16, 16, 1, 1], output_shape=[5, 150, 200, 1], stride=8,name='stem5_2_to_stem2_1')
    stem5_2_to_stem1_1 = deconv2d(seg5_3, f_shape=[32, 32, 1, 1], output_shape=[5, 300, 400, 1], stride=16,name='stem5_2_to_stem1_1')
    
    
    stem3_1 = stem3*stem5_2_to_stem3_1
    stem2_1 = stem2*stem5_2_to_stem2_1
    stem1_1 = stem1*stem5_2_to_stem1_1
    
    stem3_1 = conv2d_block(x=stem3_1, c=16, k=3, s=1, is_train=training, BR=True, name='stem3_1')
    stem2_1 = conv2d_block(x=stem2_1, c=16, k=3, s=1, is_train=training, BR=True, name='stem2_1')
    stem1_1 = conv2d_block(x=stem1_1, c=16, k=3, s=1, is_train=training, BR=True, name='stem1_1')
    
    ## stem3_1_to_3_2 = conv2d_block(x=stem3_1, c=16, k=3, s=1, is_train=training, BR=True, name='stem3_1_to_3_2')
    stem2_1_to_3_2 = conv2d_block(x=stem2_1, c=16, k=1, s=2, is_train=training, BR=True, name='stem2_1_to_3_2')
    # stem3_2_concat = tf.concat([stem3_1_to_3_2, stem2_1_to_3_2], axis=-1, name='stem3_2_concat')
    stem3_2_concat = tf.concat([stem3_1, stem2_1_to_3_2], axis=-1, name='stem3_2_concat')
    stem3_2 = conv2d_block(x=stem3_2_concat, c=16, k=3, s=1, is_train=training, BR=True, name='stem3_2')
    stem3_3 = conv2d_block(x=stem3_2, c=1, k=3, s=1, is_train=training, BR=False, name='stem3_3')
    
    
    stem1_1_to_2_2 = conv2d_block(x=stem1_1, c=16, k=1, s=2, is_train=training, BR=True, name='stem1_1_to_2_2')
   ## stem2_1_to_2_2 = conv2d_block(x=stem2_1, c=16, k=3, s=1, is_train=training, BR=True, name='stem2_1_to_2_2')
    stem3_1_to_2_2 = deconv2d(stem3_1, f_shape=[4, 4, 16, 16], output_shape=[5, 150, 200, 16], stride=2,name='stem3_1_to_2_2')
    #stem2_2_concat = tf.concat([stem1_1_to_2_2, stem2_1_to_2_2, stem3_1_to_2_2], axis=-1, name='stem2_2_concat')
    stem2_2_concat = tf.concat([stem1_1_to_2_2, stem2_1, stem3_1_to_2_2], axis=-1, name='stem2_2_concat')
    # stem2_2 = conv2d_block(x=stem2_2_concat, c=1, k=3, s=1, is_train=training, BR=False, name='stem2_2')
    stem2_2 = conv2d_block(x=stem2_2_concat, c=16, k=3, s=1, is_train=training, BR=True, name='stem2_2')
    stem2_3 = conv2d_block(x=stem2_2, c=1, k=3, s=1, is_train=training, BR=False, name='stem2_3')

   ## stem1_1_to_1_2 = conv2d_block(x=stem1_1, c=16, k=3, s=1, is_train=training, BR=True, name='stem1_1_to_1_2')
    stem2_1_to_1_2 = deconv2d(stem2_1, f_shape=[4, 4, 16, 16], output_shape=[5, 300, 400, 16], stride=2,name='stem2_1_to_1_2')
    # stem1_2_concat = tf.concat([stem2_1_to_1_2, stem1_1_to_1_2], axis=-1, name='stem1_2_concat')
    stem1_2_concat = tf.concat([stem2_1_to_1_2, stem1_1], axis=-1, name='stem1_2_concat')
    stem1_2 = conv2d_block(x=stem1_2_concat, c=16, k=3, s=1, is_train=training, BR=True, name='stem1_2')
    stem1_3 = conv2d_block(x=stem1_2_concat, c=1, k=3, s=1, is_train=training, BR=False, name='stem1_3')

    

   
    
    # stem3_2_up = deconv2d(stem3_2, f_shape=[16, 16, 1, 1], output_shape=[5, 300, 400, 1], stride=8, name='stem3_2_up')
    # stem2_2_up = deconv2d(stem2_2, f_shape=[8, 8, 1, 1], output_shape=[5, 300, 400, 1], stride=4, name='stem2_2_up')
    # stem1_2_up = deconv2d(stem1_2, f_shape=[4, 4, 1, 1], output_shape=[5, 300, 400, 1], stride=2, name='stem1_2_up')
    stem3_3_up = deconv2d(stem3_3, f_shape=[8, 8, 1, 1], output_shape=[5, 300, 400, 1], stride=4, name='stem3_3_up')
    stem2_3_up = deconv2d(stem2_3, f_shape=[4, 4, 1, 1], output_shape=[5, 300, 400, 1], stride=2, name='stem2_3_up')
    stem1_3_up = stem1_3
   
    # stemfuse_concat = tf.concat([stem3_2_up, stem2_2_up, stem1_2_up], axis=-1, name='stemfuse_concat')
    stemfuse_concat = tf.concat([stem3_3_up, stem2_3_up, stem1_3_up], axis=-1, name='stemfuse_concat')
    stemfuse =  tf.layers.conv2d(stemfuse_concat, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                    name='stemfuse', kernel_initializer=tf.constant_initializer(
                                    0.33))
    return stem1_3_up, stem2_3_up, stem3_3_up, stemfuse



