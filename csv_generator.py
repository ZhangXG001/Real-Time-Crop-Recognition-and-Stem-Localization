#coding:utf-8
import os
import csv    # 导入csv模块，主要用于csv文件操作

# 创建csv文件
def create_csv(dirname):    
    path = './dataset/'+ dirname +'/'    # 保存文件的路径
    name = os.listdir(path)    # 获取指定的路径下的文件名列表
    name.sort(key=lambda x: (x.split('_')[0][-1:]))
    #print(name)
    with open (dirname+'.csv','w', newline='') as csvfile:    # 打开路径下的csv文件
#    with open (dirname+'.csv','w') as csvfile:    # 打开路径下的csv文件

        writer = csv.writer(csvfile)    # 创建writer
        for n in name:
            if n[-4:] == '.jpg':    #判断是否jpg文件
                print(n)    # 打印输出文件名
                # with open('data_'+dirname+'.csv','rb') as f:
                writer.writerow(['./dataset/'+str(dirname) +'/'+ str(n),'./dataset/' + str(dirname) + 'label/' + str(n[:-4] + '.png')])    # 写入包含路径的原文件名和加PNG的文件名，方便model中读取文件
            else:
                pass

if __name__ == "__main__":     # 判断是否在运行本模块的程序
    create_csv('train')    # 创建train.csv文件
    create_csv('validation')    # 创建validation.csv文件
