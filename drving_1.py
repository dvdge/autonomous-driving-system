'''
无人驾驶仿真实训项目
无人驾驶项目
完成工作：完成自动驾驶
'''
#1、导入第三方库
#(1)互联网通信类
import socketio #停车场
import eventlet.wsgi #收费
from flask import Flask #管理
#(2)图像处理类
import base64,cv2
from io import BytesIO
from PIL import Image
import numpy as np

#2、设置初始化变量
max_speed = 15# 设置最大速度为 15
steering_angle = -0.02# 初始转向角度设置为 -0.02
throttle = 0.3# 初始油门设置为 0.3
def send_control(steering_angle,throttle):
    sio.emit('steer',data = {
        'steering_angle':steering_angle.__str__(), # 将转向角度转换为字符串发送
        'throttle':throttle.__str__()# 将油门转换为字符串发送
    })
#3、创建网络连接
sio=socketio.Server()#创建sio对象，定义socketio服务
app=Flask(__name__)#采用Flask做应用程序
app=socketio.WSGIApp(sio,app)#连接服务器和应用，将服务器和应用打包成WSGI
#4、传递参数，控制汽车运行
@sio.on('connect')
def on_connect(sid,environ):
    print('successfully connected')# 打印成功连接的信息
@sio.on('telemetry')#事件的通信
def on_telemetry(sid,data):
    if data:
        #print(data)
        speed = float(data['speed'])
        #print('speed',speed)
        image = Image.open(BytesIO(base64.b64decode(data['image'])))# 将 base64 编码的图像数据解码并打开为图像
        image = np.array(image)# 将图像转换为 numpy 数组，方便后续处理
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)# 转换图像颜色空间，从 BGR 转为 RGB
        cv2.imshow('Image from Udacity Simulator',image) # 显示处理后的图像
        cv2.waitKey(1)

        throttle = 1.0 - steering_angle**2 - (speed/max_speed)**2
        send_control(steering_angle,throttle)# 发送计算后的转向角度和油门值到模拟器，控制车辆行驶
    else:
        sio.emit('manual',data={})

@sio.on('disconnect')
def on_disconnect(sid):
    print('lose connected')# 打印失去连接的信息


#5、启动运行
eventlet.wsgi.server(eventlet.listen(('',4567)),app)
#使用WSGI服务器托管运行监听4567端口