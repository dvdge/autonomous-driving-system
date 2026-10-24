'''无人驾驶仿真实训项目
无人驾驶项目
'''
# 1、导入第三方库
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
Sklearn (全称 Scikit-Learn) 是基于 Python 语言的机器学习工具。把机器学习中的常用模块如分类、回归、聚类、降维、
模型选择和预处理等， 做成API ，使用简单，很适合新手上路。常与NumPy, Pandas 和 Matplotlib 一同使用。
'''

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from preprocessing_2 import image_height, image_width, image_channels
from preprocessing_2 import image_preprocessing, image_normalized
from build_model import build_model1,build_model2,build_model3

# 2、初始化变量
data_path = 'data_lake/'
test_ration = 0.1
batch_size = 100
batch_num = 200
epoch = 50

# 3、导入数据
def load_data(data_path):
    data_csv = pd.read_csv(data_path+'driving_log.csv',names=['center','left','right','steering',
                            '_','__','___'])#names:为数据加标注
    # print(data_csv)
    X=data_csv[['center','left','right']].values#有监督学习，将数据分为“输入”+“标签”
    # print(X)
    Y=data_csv['steering'].values#标签：期望的输出值
    # print(Y)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_ration,random_state=0)
    #X:数据（输入）；Y:标签（输出）分成训练集和测试集
    # print(X_train,X_test,Y_train,Y_test)
    return X_train,X_test,Y_train,Y_test
# 4、创建数据生成器（喂料机）
def batch_generator(data_path,batch_size,X_data,Y_data,train_flag):
    image_container=np.empty([batch_size,image_height, image_width, image_channels])#定义容器，盛放数据
    steer_container=np.empty(batch_size)
    while True:
        ii=0
        for index in np.random.permutation(X_data.shape[0]):#range(),np.random.choice()不同
            center,left,right=data_path+X_data[index]
            steering_angle=Y_data[index]
            if train_flag and np.random.rand()<0.6:
                image,steering_angle=image_preprocessing(center,left,right,steering_angle)
            else:
                image=cv2.imread(center)
            image_container[ii]=image_normalized(image)
            steer_container[ii]=steering_angle
            ii+=1
            if ii==batch_size:
                break
        yield image_container,steer_container

# 5、训练模型
X_train,X_test,Y_train,Y_test=load_data(data_path)
model=build_model2()
checkpoint = ModelCheckpoint(
    'lake_model2_{epoch:03d}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)
stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=200,
    verbose=1,
    mode='auto'
)
tensor_board = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=1,
    write_images=0
)
model.compile(optimizer=Adam(learning_rate=0.0001),loss='mse',metrics=['accuracy'])
model.fit(
    batch_generator(data_path,batch_size,X_train,Y_train,True),
    steps_per_epoch=batch_num,
    epochs=epoch,
    verbose=1,
    validation_data=batch_generator(data_path,batch_size,X_test,Y_test,False),
    validation_steps=1,
    max_queue_size=1,
    callbacks=[checkpoint,stopping,tensor_board]
)

# 6、保存模型
model.save('lake_model2.h5')

