import tensorflow as tf
import os
import numpy as np


def set_config():
    '''
    set the GPU options
    '''

    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    config = tf.ConfigProto(gpu_options=gpu_options)
    session = tf.Session(config=config)


def conv2d_block(x,c,k,s,is_train,BR,name):
    x = conv2d(input_=x, output_dim=c, k_h=k, k_w=k, d_h=s, d_w=s, stddev=0.02, name=name, bias=False)
    #x= tf.layers.conv2d(x,filters=c,kernel_size=[k,k],strides=s,padding='SAME')
    if (BR == True):
        x = batch_norm(x, momentum=0.9, epsilon=1e-5, train=is_train, name=name)
        x = tf.nn.relu(x)
    return x

def get_deconv_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(initializer=init, name="up_filter",
                                       shape=weights.shape, trainable=False)  # ,,trainable=False
    return bilinear_weights

def deconv2d(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  # sess_temp = tf.InteractiveSession()
  sess_temp = tf.global_variables_initializer()
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape,stride)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv


def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      training=train,
                      name=name)

def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(1e-4),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)



def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv(input, bottleneck_dim, name='pw/1', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


def get_residual_layer(res_n) :
    x = []
    if res_n == 10 :
        x = [2,2,2,2]
    return x


def train_ops(loss, learning_rate):
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss, global_step=global_step)




def data_augmentation(image, label_seg, label_stem, training=True):
    "image augmentation including random flip"
    #mean, variance = tf.nn.moments(image, axes=[0, 1, 2], keep_dims=True)
    #image = (image - mean) / variance
    #image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image)) - 0.5

    if training:
        image_label = tf.concat([image, label_seg, label_stem], axis=-1)
        print('image label shape concat', image_label.get_shape())
        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
        image = maybe_flipped[:, :, :-2]
        label_seg = maybe_flipped[:, :, -2:-1]
        label_stem = maybe_flipped[:, :, -1:]
        return image, label_seg, label_stem


def read_csv(queue, augmentation=True):
    "read the image,stem label and plant label"
    csv_reader = tf.TextLineReader(skip_header_lines=1)

    _, csv_content = csv_reader.read(queue)

    image_path, label_seg_path, label_stem_path = tf.decode_csv(csv_content, record_defaults=[[""], [
        ""], [""]])

    image_file = tf.read_file(image_path)
    label_seg_file = tf.read_file(label_seg_path)
    label_stem_file = tf.read_file(label_stem_path)

    image = tf.image.decode_jpeg(image_file, channels=3)
    image.set_shape([300, 400, 3])
    image = tf.cast(image, tf.float32)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image)) - 0.5
    print('image shape', image.get_shape())

    label_seg = tf.image.decode_png(label_seg_file, channels=1)
    label_seg.set_shape([300, 400, 1])

    label_stem = tf.image.decode_png(label_stem_file, channels=1)

    label_seg = tf.cast(label_seg, tf.float32)
    label_stem = tf.cast(label_stem, tf.float32)

    label_seg = label_seg / (
        tf.reduce_max(label_seg))
    label_stem = label_stem / (
        tf.reduce_max(label_stem))

    if augmentation:
        image, label_seg, label_stem = data_augmentation(image, label_seg,
                                                         label_stem)
    else:
        pass
    return image, label_seg, label_stem


def loss_CE1(logits, labels):
    loss_weight = np.array([
        1.0,
        3.0
    ])
    labels = tf.to_int64(labels)
    loss = weighted_loss(logits, labels, number_class=2, frequency=loss_weight)
    return loss


def weighted_loss(logits, labels, number_class, frequency):
    """
    The reference paper is : https://arxiv.org/pdf/1411.4734.pdf 
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    we weight each pixels by alpha_c
    Inputs: 
    logits is the output from the inference, which is the output of the decoder layers without softmax.
    labels: true label information 
    number_class: In the  dataset, it's 2 classes
    Outputs:
    Loss
    Accuracy
    """
    
    #label_flatten = tf.reshape(labels, [-1])
    label_onehot = tf.one_hot(label_flatten, depth=number_class)
    logits = tf.nn.sigmoid(logits)
    logits = tf.concat([1-logits,logits],axis=-1)
    logits_reshape = tf.reshape(logits, [-1, number_class])
    #logits_reshape = tf.nn.softmax(logits_reshape)
    #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot, logits=logits_reshape,
                                                          #   pos_weight=frequency)
    cross_entropy =  tf.nn.softmax_cross_entropy_with_logits (labels=label_onehot,
                                                            logits=logits_reshape)
    cross_entropy=tf.multiply(cross_entropy, frequency)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean



def loss_CE(logits,labels,weight=1.0):
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,pos_weight=weight)
    cost = tf.reduce_mean(cross_entropy)
    return cost


def loss_CE1(y_pred, y_true):
    '''flat_logits = tf.reshape(y_pred,[-1,num_class])
    flat_labels = tf.reshape(y_true,[-1,num_class])
    class_weights = tf.constant(loss_weight,dtype=np.float32)
    weight_map = tf.multiply(flat_labels,class_weights)
    weight_map = tf.reduce_sum(weight_map,axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels,logits=flat_logits)

    #weighted_loss = tf.multiply(loss_map,weight_map)

    #cross_entropy_mean = tf.reduce_mean(weighted_loss)
    cross_entropy_mean = -tf.reduce_mean(tf.reduce_sum(y_true*tf.log(y_pred)))'''

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)  # 计算logits和labels每个对应维度上对应元素的损失
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算各维度上所有元素上的平均值

    return cross_entropy_mean
    
def loss_IOU(y_pred, y_true):
    y_pred = tf.nn.sigmoid(y_pred)
    n_class = int(y_true.get_shape().as_list()[-1])
    Iou = []
    for class_ in range(n_class):
        i = tf.reduce_sum(y_pred[:, :, :, class_]*y_true[:, :, :, class_])
        u = tf.reduce_sum(y_pred[:, :, :, class_]+y_true[:, :, :, class_] -
                   (y_pred[:, :, :, class_]*y_true[:, :, :, class_]))
        iou = i/u
        Iou.append(iou)
    return 1-tf.reduce_mean(Iou)

# 计算交叉熵损失
def loss_MSE(y_pred, y_true):
    mse = tf.losses.mean_squared_error(y_true, y_pred)
    return mse

def pooling(input, s):
    return tf.layers.average_pooling2d(
    inputs = input,
    pool_size = 1,
    strides = s,
    padding='same'
    )
