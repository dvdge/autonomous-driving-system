'''无人驾驶仿真实训项目
无人驾驶项目
完成工作：1。数据扩充————解决数据量不够的问题
        2.平衡数据类型————让模型“见多识广”
        3.数据归一化————提高模型效率'''
#1.导入第三方库
import cv2
import numpy as np
#2.设置初始化变量
image_height,image_width,image_channels=66,200,3
center,left,right='test/center.jpg','test/left.jpg','test/right.jpg'
steering_angle=0.0
#3.选择图像
def image_choose(center,left,right,steering_angle):
    choice=np.random.choice(3) #在0,1,2中随机选择数字，随机选择图像
    if choice==0:
        image_name=center
        bias=0.0
    if choice==1:
        image_name=left
        bias=0.2
    if choice==2:
        image_name=right
        bias=-0.2
    image=cv2.imread(image_name)
    #cv2.imshow('image_choose',image)
    #cv2.waitKey(0)
    steering_angle=steering_angle+bias
    return image,steering_angle
#4.平移图像
def image_translate(image,steering_angle):
    range_x,range_y=100,10
    tran_x=int(range_x*(np.random.rand()-0.5))
    tran_y=int(range_y*(np.random.rand()-0.5))
    tran_m=np.float32([[1,0,tran_x],[0,1,tran_y]])
    image=cv2.warpAffine(image,tran_m,(image.shape[1],image.shape[0]))
    steering_angle=steering_angle+tran_x*0.002
    #cv2.imshow('image_translate',image)
    #cv2.waitKey(0)
    return image,steering_angle
#5.翻转图像
def image_flip(image,steering_angle):
    if np.random.rand()<0.5:
        image=cv2.flip(image,1) #1水平翻转；0垂直翻转；-1水平垂直同时翻转
        steering_angle=-steering_angle
    #cv2.imshow('image_flip',image)
    #cv2.waitKey(0)
    return image,steering_angle

#6.归一化图像
def image_normalized(image):
    image=image[60:-25,:,:] #对图片进行裁剪，去除一些无关紧要的因素。例如：天空
    image=cv2.resize(image,(image_width,image_height),cv2.INTER_AREA)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
    #cv2.imshow('image_normalized',image)
    #cv2.waitKey(0)
    return image

#7.定义图像预处理函数
#for i in range(5):
def image_preprocessing(center,left,right,steering_angle):
    image,steering_angle = image_choose(center,left,right,steering_angle)
    image,steering_angle = image_translate(image,steering_angle)
    image,steering_angle = image_flip(image,steering_angle)
    #image = image_normalized(image)
    return image,steering_angle
#8.设置主函数运行
if __name__=='__main__':
    image,steering_angle=image_preprocessing(center,left,right,steering_angle)
    image = image_normalized(image)
    cv2.imshow('image_data',image)
    cv2.waitKey(0)
    print(steering_angle)
    cv2.destroyAllWindows()

