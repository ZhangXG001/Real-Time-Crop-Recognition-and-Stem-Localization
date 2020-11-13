import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import cv2
import time
from ops import train_ops, set_config, data_augmentation, read_csv, loss_CE, loss_IOU, loss_MSE, pooling
from model import Network
from tensorflow.python.framework import graph_util
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default='./train.csv')

parser.add_argument('--validation_dir',
                    default='./validation.csv')

parser.add_argument('--model_dir',
                    default='./modelstem3')

parser.add_argument('--epochs',
                    type=int,
                    default=250)

parser.add_argument('--peochs_per_eval',
                    type=int,
                    default=1)

parser.add_argument('--logdir',
                    default='./logsstem2')

parser.add_argument('--batch_size',
                    type=int,
                    default=5)

parser.add_argument('--is_cross_entropy',
                    action='store_true',
                    default=True)

parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-4)

parser.add_argument('--decay_rate',
                    type=float,
                    default=0.1)

parser.add_argument('--decay_step',
                    type=int,
                    default=12000)

parser.add_argument('--weight',
                    nargs='+',
                    type=float,
                    default=[1.0, 1.0])
 
parser.add_argument('--random_seed',
                    type=int,
                    default=1234)

parser.add_argument('--gpu',
                    type=str,
                    default=2)

flags = parser.parse_args()


def main(flags):
    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(flags.logdir, "train", current_time)
    validation_logdir = os.path.join(flags.logdir, "validation", current_time)

    train = pd.read_csv(flags.data_dir)
    num_train = train.shape[0]

    validation = pd.read_csv(flags.validation_dir)
    num_validation = validation.shape[0]

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[flags.batch_size, 300, 400, 3], name='X')
    y_seg = tf.placeholder(tf.float32, shape=[flags.batch_size, 300, 400, 1], name='y_seg')
    y_stem = tf.placeholder(tf.float32, shape=[flags.batch_size, 300, 400, 1], name='y_stem')  
    training = tf.placeholder(tf.bool, name='training')
    # seg1_2, seg3_2, seg5_2, segfuse, stem1_2, stem2_2, stem3_2, stemfuse = Network(X, training=training)
    seg1_3, seg3_3, seg5_3, segfuse, stem1_3, stem2_3, stem3_3, stemfuse = Network(X, training=training)
    
   
    loss_segfuse = loss_CE(segfuse, y_seg)
    # loss_seg1_2 = loss_CE(seg1_2, y_seg)
    # loss_seg3_2 = loss_CE(seg3_2, y_seg)
    # loss_seg5_2 = loss_CE(seg5_2, y_seg)
    loss_seg1_3 = loss_CE(seg1_3, y_seg)
    loss_seg3_3 = loss_CE(seg3_3, y_seg)
    loss_seg5_3 = loss_CE(seg5_3, y_seg)
    
    
    loss_stemfuse = loss_IOU(stemfuse, y_stem)
    # loss_stem1_2 = loss_CE(stem1_2, y_stem)
    # loss_stem2_2 = loss_CE(stem2_2, y_stem)
    # loss_stem3_2 = loss_CE(stem3_2, y_stem)
    loss_stem1_3 = loss_IOU(stem1_3, y_stem)
    loss_stem2_3 = loss_IOU(stem2_3, y_stem)
    loss_stem3_3 = loss_IOU(stem3_3, y_stem)
    
    
    # Sum all loss terms.
    # total_loss = loss_stem1_3+loss_stem2_2+loss_stem3_2+loss_stemfuse +loss_seg1_2+loss_seg3_2+loss_seg5_2+loss_segfuse
    total_loss =  loss_stem1_3+loss_stem2_3+loss_stem3_3+loss_stemfuse

    tf.summary.scalar("total_loss", total_loss)
    

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                               decay_steps=flags.decay_step,
                                               decay_rate=flags.decay_rate,
                                               staircase=True)
                                               
    tf.summary.scalar("loss_segfuse",  loss_segfuse)
    # tf.summary.scalar("loss_seg1_2",  loss_seg1_2)
    # tf.summary.scalar("loss_seg3_2",  loss_seg3_2)
    # tf.summary.scalar("loss_seg5_2",  loss_seg5_2) 
    tf.summary.scalar("loss_seg1_3",  loss_seg1_3)
    tf.summary.scalar("loss_seg3_3",  loss_seg3_3)
    tf.summary.scalar("loss_seg5_3",  loss_seg5_3) 
    
    tf.summary.scalar("loss_stemfuse",  loss_stemfuse)
    # tf.summary.scalar("loss_stem1_2", loss_stem1_2)
    # tf.summary.scalar("loss_stem2_2", loss_stem2_2)
    # tf.summary.scalar("loss_stem3_2", loss_stem3_2)
    tf.summary.scalar("loss_stem1_3", loss_stem1_3)
    tf.summary.scalar("loss_stem2_3", loss_stem2_3)
    tf.summary.scalar("loss_stem3_3", loss_stem3_3)
    
    tf.summary.scalar("learning_rate", learning_rate)
    
    
    img_segfuse = tf.nn.sigmoid(segfuse)
    # img_seg1_2 = tf.nn.sigmoid(seg1_2)
    # img_seg3_2 = tf.nn.sigmoid(seg3_2)
    # img_seg5_2 = tf.nn.sigmoid(seg5_2)
    img_seg1_3 = tf.nn.sigmoid(seg1_3)
    img_seg3_3 = tf.nn.sigmoid(seg3_3)
    img_seg5_3 = tf.nn.sigmoid(seg5_3)
    
    img_stemfuse = tf.nn.sigmoid(stemfuse)
    # img_stem1_2 = tf.nn.sigmoid(stem1_2)
    # img_stem2_2 = tf.nn.sigmoid(stem2_2)
    # img_stem3_2 = tf.nn.sigmoid(stem3_2)
    img_stem1_3 = tf.nn.sigmoid(stem1_3)
    img_stem2_3 = tf.nn.sigmoid(stem2_3)
    img_stem3_3 = tf.nn.sigmoid(stem3_3)
    
    tf.summary.image("img_segfuse", img_segfuse)
    # tf.summary.image("img_seg1_2",  img_seg1_2)
    # tf.summary.image("img_seg3_2",  img_seg3_2)
    # tf.summary.image("img_seg5_2",  img_seg5_2)
    tf.summary.image("img_seg1_3",  img_seg1_3)
    tf.summary.image("img_seg3_3",  img_seg3_3)
    tf.summary.image("img_seg5_3",  img_seg5_3)
    
    tf.summary.image("img_stemfuse", img_stemfuse)
    # tf.summary.image("img_stem1_2", img_stem1_2)
    # tf.summary.image("img_stem2_2", img_stem2_2)
    # tf.summary.image("img_stem3_2", img_stem3_2)
    tf.summary.image("img_stem1_3", img_stem1_3)
    tf.summary.image("img_stem2_3", img_stem2_3)
    tf.summary.image("img_stem3_3", img_stem3_3)
    
    tf.summary.image('Input Image:', X)
    tf.summary.image('Label_seg:', y_seg)
    tf.summary.image('Label_stem:', y_stem)
    
    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', training)
    tf.add_to_collection('seg', img_segfuse)
    tf.add_to_collection('stem',img_stemfuse) 

    summary_op = tf.summary.merge_all()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op =train_ops(total_loss,learning_rate)
        

   
    train_csv = tf.train.string_input_producer(['train.csv'])
    validation_csv = tf.train.string_input_producer(['validation.csv'])

    train_image, train_seg_label, train_stem_label = read_csv(train_csv, augmentation=True)
    validation_image, validation_seg_label, validation_stem_label = read_csv(validation_csv, augmentation=True)
    
    X_train_batch_op, y_train_seg_batch_op, y_train_stem_batch_op = tf.train.shuffle_batch([train_image, train_seg_label, train_stem_label], batch_size=flags.batch_size,

                                                                capacity=flags.batch_size * 500,
                                                                min_after_dequeue=flags.batch_size * 100,

                                                                allow_smaller_final_batch=True)

    X_validation_batch_op, y_validation_seg_batch_op, y_validation_stem_batch_op = tf.train.batch([validation_image, validation_seg_label, validation_stem_label], batch_size=flags.batch_size,

                                                      capacity=flags.batch_size * 20, allow_smaller_final_batch=True)

    print('Shuffle batch done')

    

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        validation_writer = tf.summary.FileWriter(
            validation_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        try:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            a=[]
            for epoch in range(flags.epochs):
                
                for step in range(0, num_train, flags.batch_size):
                    start_time=time.time()
                    X_train, y_seg_train, y_stem_train = sess.run([X_train_batch_op, y_train_seg_batch_op, y_train_stem_batch_op])  
                    _, step_ce, step_summary, global_step_value = sess.run([train_op, total_loss, summary_op, global_step],
                                                                           feed_dict={X: X_train, y_seg: y_seg_train, y_stem: y_stem_train, 
                                                                                      training: True})
                    duration=time.time()-start_time
                    a.append(duration)
                    if (global_step_value % 100 ==0):
                        train_writer.add_summary(step_summary, global_step_value)
                    print(
                    'epoch:{} step:{} loss_CE:{}'.format(epoch + 1, global_step_value, step_ce))
                    
                for step in range(0, num_validation, flags.batch_size):
                    X_test, y_seg_test,y_stem_test = sess.run([X_validation_batch_op, y_validation_seg_batch_op,y_validation_stem_batch_op]) 
                    step_ce, step_summary = sess.run([total_loss, summary_op], feed_dict={X: X_test, y_seg: y_seg_test, y_stem:y_stem_test,
                                                                                    training: False})

                    validation_writer.add_summary(step_summary, epoch * (
                        num_train // flags.batch_size) + step // flags.batch_size * num_train // num_validation)
                    print('Test loss_CE:{}'.format(step_ce))
                saver.save(sess, '{}/model.ckpt'.format(flags.model_dir))
                
            train_time_batch=np.mean([a])
            print('train time per batch:{}'.format(train_time_batch))
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.model_dir))


if __name__ == '__main__':
    set_config()
    main(flags)
