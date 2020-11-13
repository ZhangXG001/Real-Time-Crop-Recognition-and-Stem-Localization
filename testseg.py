# coding:utf-8
# Bin GAO

# 加了batch_size设置
# 注意：务必保证测试图像的数量是batch size的整数倍
# 2018.11.07 修改了输出图像的保存格式，由原来的与输入图像相同改为png 


import os
import cv2    # 导入openCV模块
import glob    # 导入glob模块，主要用于查找文件
import tensorflow as tf
import numpy as np
import argparse    # 导入argparse模块，用于解析命令行参数和选项
import time
ep=1e-8

batch_size = 5  # 批大小，需要根据训练时设置的批大小修改

# 这部分代码的注释参照train.py中的
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    type=str,
                    default=r'/home/AirsZXG/StemNet-V1.1/dataset/test/')
parser.add_argument('--model_dir',
                    type=str,
                    default=r'./modelseg1(160)')#./model1
parser.add_argument('--save_dir',
                    type=str,
                    default=r'./resultseg1(160)')#./result1
parser.add_argument('--gpu',
                    type=int,
                    default=0)
flags = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # 只允许第一个GPU可见（只用第一个GPU）



# 加载模型
def load_model():
    file_meta = os.path.join(flags.model_dir, 'model.ckpt.meta')    # 把字符串组合成路径，里面保存了计算图（即构建的网络）的结构
    file_ckpt = os.path.join(flags.model_dir, 'model.ckpt')    # 把字符串组合成路径，里面保存了模型参数

    saver = tf.train.import_meta_graph(file_meta)    #    导入计算图结构
    # tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

    sess = tf.InteractiveSession()    # 创建交互式session，可自动指定为默认session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)  
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver.restore(sess, file_ckpt)    # 恢复最后一次保存的模型参数
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
    return sess

# 调用OpenCV库函数读取图像
def read_image(image_path, gray=False):
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    # 若指定读入图像为灰度图，则把该图像作为返回值
    else:
        image = cv2.imread(image_path)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # 若指定读入彩色图像，则转换通道顺序后作为返回值

# 主入口函数
def main(flags):
    sess = load_model()    # 加载模型
    
    X = tf.get_collection('inputs')[0]    # 从集合'inputs'中读取输入数据
#    print('shape of X', X.get_shape())
    training = tf.get_collection('inputs')[1]    # 从集合'inputs'中读取输入数据
    pred = tf.get_collection('seg')[0]    # 从集合'upscore_fuse'中读取第一个元素，也就是side output融合层的输出
    '''
    pred_f = tf.get_collection('upscore_fuse')[0] 
    pred_2 = tf.get_collection('score_dsn2_up')[0] 
    pred_3 = tf.get_collection('score_dsn3_up')[0] 
    pred_4 = tf.get_collection('score_dsn4_up')[0] 
    loss_f2 =tf.add(pred_f,pred_2)
    loss_f23=tf.add(loss_f2,pred_3)
    loss_f234=tf.add(loss_f23,pred_4)
    out=loss_f234/4
    pred=(out-tf.reduce_min(out)+ep)/(tf.reduce_max(out)-tf.reduce_min(out)+ep)
    '''
    
#    print('shape of pred', pred.get_shape())
    
    names=os.listdir(flags.input_dir)    # 获取指定的路径下的文件名列表
    
    num_images = 0     # 用于记录读入图像数量
    num_batchs = 0    # 用于记录读入图像的批数
    image_feed = []    # 用于把图像拼接成batch
    a=[]
    
    # names.remove('.DS_Store')
    for name in names:
        inputname=os.path.join(flags.input_dir,name)    # 组合成输入图像路径
        image = read_image(inputname)    # 读取图像
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) - 0.5
       # image=cv2.resize(image,(1600,1200))    #把图像缩放指定尺寸
#        image=cv2.resize(image,(400,300))    # 尺寸参数是[宽，高]
        # sess=tf.InteractiveSession()
        
#        image = np.expand_dims(image, 0)    # 增加一个维度
        num_images += 1    # 每读入一张图片自增1        
#        if image_feed == []:    # 若image_feed被清空，则把读入的图像数据赋给他
        image_feed.append(image)
#        else:
#            image_feed = tf.concat([image_feed,image],axis = 0)    # 把图像在第一维连接起来
#        print(image_feed.get_shage())

        # 累积一个batch sizeh后，导入模型进行计算，并保存处理后的图片
        if num_images == batch_size:
            start2=time.time()
            label_pred = sess.run(pred, feed_dict={X: image_feed, training: False})    # 喂入数据，计算输出
            end2=time.time()
            print(end2-start2)
            a.append(end2-start2)
            
            print('label_pred的形状为 ',label_pred.shape)
            
            for i in range(label_pred.shape[0]):
                merged = np.squeeze(label_pred[i])    #删除label_pred张量中长度为1的维度，即变成一个二维矩阵
                merged=np.uint8((merged)*255)
                #merged=np.uint8((merged+1)*127.5)
                #_, merged = cv2.threshold(merged, 170, 255, cv2.THRESH_BINARY)    # 将值大于127的像素设置为白色（255）
                crt_name = names[num_batchs*batch_size+i]    # 获取当前图片的文件名
                save_name = os.path.join(flags.save_dir, crt_name[:-4]+".png")    # 组合成保存处理结果的路径
                cv2.imwrite(save_name, merged)    # 保存图片到直径路径下
                print('Pred saved')
            
            image_feed = []    # 清空
            num_images = 0    # 计数器归零
            num_batchs += 1    # 每读入一批图片自增1
    b=np.mean(a[2:]) 
    print(b)
if __name__ == '__main__':    # 判断是否在运行本模块的程序
    main(flags)    # 如果上一句判断为true,则说明在直接运行本模块，那就执行main()函数

