'''
无人驾驶仿真实训项目
项目一 基于深度学习的 Udacity 无人驾驶系统
'''
#完成工作：完成自动驾驶
# 1、导入第三方库
#   (1)互联网通信类
import socketio  # 导入库
import eventlet.wsgi  #
from flask import Flask
#   （2）图像处理类
import base64, cv2
from io import BytesIO
from PIL import Image
import numpy as np

#    (3)模型相关类
from tensorflow.keras.models import load_model
from preprocessing_2 import image_normalized
model=load_model('mountain_model2.h5')
# 2、初始化变量及函数
class SimplePIControl:
    def __init__(self, KP, KI):
        self.KP = KP
        self.KI = KI
        self.error = 0.0
        self.set_point = 0.0
        self.integral = 0.0
        self.throttle = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def updated(self, measurement):
        self.error = self.set_point - measurement
        self.integral += self.error
        self.throttle = self.error * self.KP + self.integral * self.KI
        return self.throttle


controller = SimplePIControl(0.1, 0.002)
set_speed = 25
controller.set_desired(set_speed)
steering_angle = -0.02
throttle = 0.3


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

# 3、创建网络连接
sio = socketio.Server()
app = Flask(__name__)
app = socketio.WSGIApp(sio, app)


# 4、传递参数，控制汽车行驶
@sio.on('connect')
def on_connect(sid, environ):
    print('与模拟器连接成功！')


@sio.on('telemetry')
def on_telemetry(sid, data):
    if data:
        # print('收到信息',data)
        speed = float(data['speed'])
        # print('speed', speed)
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Image from Udacity Simulator', image)
        cv2.waitKey(1)
        #print(image)
        #throttle=1.0-steering_angle**2-(speed/set_speed)**2
        image=image_normalized(image)
        steering_angle=float(model.predict(np.array([image])))
        throttle=controller.updated(speed)
        send_control(steering_angle,throttle)
    else:
        sio.emit('manual',data={})


@sio.on('disconnect')
def on_disconnect(sid):
    print('与模拟器断开连接')


# 5、启动运行
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
