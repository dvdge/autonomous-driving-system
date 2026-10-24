'''
无人驾驶仿真实训项目
无人驾驶项目
完成工作：搭建卷积神经网络模型结构
'''
#1.导入第三方库
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,Lambda,MaxPool2D
from preprocessing_2 import image_height,image_width,image_channels

#2.搭建卷积神经网络模型
Input_size = (image_height,image_width,image_channels)#输入图像的长度，宽度，数据的通道数
# 构建宽度为32，深度为2的卷积神经网络
def build_model1():  # 参数数量 K*K*C*O+O（K:kernel_size,C:输入通道数，O:输出通道数）
    model = Sequential()  # 开始构建Sequential()模型结构
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))  # 把数值进行归一化（数值范围在0-1之间）：所有数据/127.5-1，提高模型效率
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
                     activation='relu'))  # 卷积层对原来的图像提取特征：24个卷积核，卷积核大小为5×5，步长为（2,2），激活函数'relu'
    model.add(MaxPool2D())  # 构建最大池化层，降低输出维度（宽高）
    model.add(Dropout(0.25))  # 按照25%的概率将一部分神经网络单元暂时从网络中丢弃
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2),
                     activation='relu'))  # 卷积层对原来的图像提取特征：32个卷积核，卷积核大小为5×5，步长为（2,2），激活函数'relu'
    model.add(MaxPool2D())  # 构建最大池化层，降低输出维度（宽高）
    model.add(Flatten())  # Flatten将多维数据扁平，转化为一维
    model.add(Dense(32))  # “降维”，输出节点数（输出的大小（维数））为32
    model.add(Dropout(0.20))  # 按照20%的概率将一部分神经网络单元暂时从网络中丢弃
    model.add(Dense(16))  # “降维”，输出节点数（输出的大小（维数））为16
    model.add(Dense(1))  # “降维”，输出节点数（输出的大小（维数））为1（最终降至一维）
    model.summary()  # summary()方法会将模型每一层的参数个数，以及整个模型参数总数和可以训练参数个数显示出来
    return model  # 返回模型model

def build_model2():
    model = Sequential()
    model.add(Lambda(lambda x:x / 127.5 - 1,input_shape = Input_size))
    model.add(Conv2D(filters=24,kernel_size=(5,5), strides=(2,2),activation='elu'))
    model.add(Conv2D(filters=36,kernel_size=(5, 5), strides=(2, 2),activation='elu'))
    model.add(Conv2D(filters=48,kernel_size=(5, 5), strides=(2, 2),activation='elu'))
    model.add(Conv2D(filters=64,kernel_size=(3, 3), strides=(1, 1),activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())#将多维数据扁平，转化为一维
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()#summary方法可以将模型每一层的参数个数
    return model

def build_model3():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=Input_size))  # 把数值进行归一化（数值范围在0-1之间）：所有数据/127.5-1，提高模型效率
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='elu'))  # 卷积层对原来的图像提取特征：32个卷积核，卷积核大小为3×3，步长为（1,1），缩小维度且四周填充0，激活函数'elu'
    # padding表示是否对周围进行填充，“same”即使通过kernel_size缩小了维度，但是四周会填充0，保持原先的维度；“valid”表示存储不为0的有效信息。
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='elu'))  # 卷积层对原来的图像提取特征：32个卷积核，卷积核大小为3×3，步长为（1,1），缩小维度且四周填充0，激活函数'elu'
    model.add(MaxPool2D((2, 2), padding='same'))  # 在 2x2 池化窗口中取最大值，并在输入的左/右或上/下均匀填充，以使输出具有与输入相同的高度/宽度尺寸
    # pool_size 整数或 2 个整数的元组，取最大值的窗口大小。 (2, 2) 将在 2x2 池化窗口中取最大值。如果只指定一个整数，则两个维度将使用相同的窗口长度。
    # padding "valid" 或 "same" 之一(不区分大小写)。 "valid" 表示没有填充。 "same" 导致在输入的左/右或上/下均匀填充，以使输出具有与输入相同的高度/宽度尺寸。
    model.add(Dropout(0.5))  # 按照50%的概率将一部分神经网络单元暂时从网络中丢弃
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='elu'))  # 卷积层对原来的图像提取特征：64个卷积核，卷积核大小为3×3，步长为（1,1），缩小维度且四周填充0，激活函数'elu'

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='elu'))  # 卷积层对原来的图像提取特征：64个卷积核，卷积核大小为3×3，步长为（1,1），缩小维度且四周填充0，激活函数'elu'
    model.add(MaxPool2D((2, 2), padding='same'))  # 在 2x2 池化窗口中取最大值，并在输入的左/右或上/下均匀填充，以使输出具有与输入相同的高度/宽度尺寸
    model.add(Dropout(0.5))  # 按照50%的概率将一部分神经网络单元暂时从网络中丢弃

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='elu'))  # 卷积层对原来的图像提取特征：32个卷积核，卷积核大小为3×3，步长为（1,1），缩小维度且四周填充0，激活函数'elu'
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                     activation='elu'))  # 卷积层对原来的图像提取特征：32个卷积核，卷积核大小为3×3，步长为（1,1），缩小维度且四周填充0，激活函数'elu'
    model.add(MaxPool2D((2, 2), padding='same'))  # 在 2x2 池化窗口中取最大值，并在输入的左/右或上/下均匀填充，以使输出具有与输入相同的高度/宽度尺寸
    model.add(Dropout(0.5))  # 按照50%的概率将一部分神经网络单元暂时从网络中丢弃
    model.add(Flatten())  # Flatten将多维数据扁平，转化为一维
    # model.add(Dense(512, activation='elu'))  # “降维”，输出节点数（输出的大小（维数））为512，激活函数'elu'
    model.add(Dense(64, activation='elu'))  # “降维”，输出节点数（输出的大小（维数））为64，激活函数'elu'
    model.add(Dense(16, activation='elu'))  # “降维”，输出节点数（输出的大小（维数））为16，激活函数'elu'
    model.add(Dense(1))  # “降维”，输出节点数（输出的大小（维数））为1（最终降至一维）
    model.summary()  # summary()方法会将模型每一层的参数个数，以及整个模型参数总数和可以训练参数个数显示出来
    return model  # 返回模型model


#3.设置主函数
if __name__ == '__main__':
    build_model1()
    build_model2()
    build_model3()