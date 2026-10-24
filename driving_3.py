# 1、导入第三方库
#   (1)互联网通信类
import socketio
import eventlet.wsgi
from flask import Flask
import time  # 用于添加延迟
#   (2)图像处理类
import base64, cv2
from io import BytesIO
from PIL import Image
import numpy as np

#   (3)模型相关类
from tensorflow.keras.models import load_model


# 2、图像处理与特征提取优化
def preprocess_image(image):
    """增强的图像预处理函数，专为环山路识别优化"""
    # 裁剪图像，去除天空和车头部分，专注于道路
    # 根据实际情况调整裁剪区域
    image = image[60:135, :, :]

    # 转换颜色空间到YUV，这对车道线检测更有效
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # 高斯模糊减少噪点
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 调整大小到模型输入尺寸
    image = cv2.resize(image, (200, 66))

    # 归一化
    image = image / 255.0

    return image


# 3、优化的PID控制器（添加微分控制）
class AdaptivePIControl:
    """自适应PID控制器，可根据转向角度动态调整参数"""

    def __init__(self, base_kp, base_ki, base_kd=0.01, steering_threshold=0.3):
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd  # 添加微分控制
        self.steering_threshold = steering_threshold
        self.error = 0.0
        self.set_point = 0.0
        self.integral = 0.0
        self.last_error = 0.0  # 用于微分计算
        self.throttle = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement, steering_angle):
        self.error = self.set_point - measurement

        # 在急转弯时增加积分项重置机制，防止积分饱和
        if abs(steering_angle) > self.steering_threshold:
            self.integral *= 0.8  # 减小积分积累

        self.integral += self.error

        # 计算微分项
        derivative = self.error - self.last_error
        self.last_error = self.error

        # 根据转向角度动态调整KP
        dynamic_kp = self.base_kp * (1 + abs(steering_angle) * 2)

        # 组合PID输出
        self.throttle = (self.error * dynamic_kp +
                         self.integral * self.base_ki +
                         derivative * self.base_kd)

        # 限制油门范围
        self.throttle = max(-1.0, min(1.0, self.throttle))

        return self.throttle


# 4、动态速度控制
def get_adaptive_speed(steering_angle, base_speed=30):
    """根据转向角度动态调整目标速度，急转弯时降低速度"""
    # 转向角度越大，速度越低
    speed_factor = max(0.3, 1.0 - abs(steering_angle) * 2)
    return base_speed * speed_factor


# 5、转向角平滑处理
def smooth_steering(new_angle, old_angle, smoothing_factor=0.3):
    """平滑转向控制，避免剧烈转向"""
    return old_angle * smoothing_factor + new_angle * (1 - smoothing_factor)


# 6、加载模型并初始化控制器
model = load_model('lake_model2.h5')
controller = AdaptivePIControl(0.1, 0.001, 0.01)  # 调整PID参数，添加KD
set_speed = 20  # 降低初始速度，更适合起步
controller.set_desired(set_speed)

# 7、全局状态变量
last_steering_angle = 0.0  # 初始转向角设为0
frame_count = 0  # 用于计数处理的帧数
start_time = None  # 记录启动时间

# 8、创建网络连接
sio = socketio.Server()
app = Flask(__name__)
app = socketio.WSGIApp(sio, app)


# 9、传递参数，控制汽车行驶
def send_control(steering_angle, throttle):
    """向模拟器发送控制命令"""
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


@sio.on('connect')
def on_connect(sid, environ):
    """连接成功回调"""
    global start_time
    print('与模拟器连接成功！')
    print('系统启动中，请等待3秒...')
    time.sleep(3)  # 添加启动延迟，让车辆有时间调整姿态
    start_time = time.time()
    print(f'自动驾驶已激活 - 初始速度: {set_speed} mph')


@sio.on('telemetry')
def on_telemetry(sid, data):
    """接收模拟器数据并发送控制命令"""
    global last_steering_angle, frame_count, start_time

    if data:
        # 获取当前速度
        speed = float(data['speed'])

        # 计算运行时间
        if start_time:
            elapsed_time = time.time() - start_time
        else:
            elapsed_time = 0

        # 处理图像
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 显示原始图像
        cv2.imshow('Original Image', image)

        # 预处理图像
        processed_image = preprocess_image(image)

        # 显示处理后的图像
        cv2.imshow('Processed Image', processed_image)
        cv2.waitKey(1)

        # 预测转向角
        steering_angle = float(model.predict(np.array([processed_image])))

        # 限制最大转向角
        steering_angle = max(-0.6, min(0.6, steering_angle))

        # 前5帧使用固定小角度（预热阶段）
        if frame_count < 5:
            steering_angle = 0.05 if steering_angle > 0 else -0.05
            frame_count += 1

        # 平滑转向角
        steering_angle = smooth_steering(steering_angle, last_steering_angle)
        last_steering_angle = steering_angle

        # 动态调整速度
        adaptive_speed = get_adaptive_speed(steering_angle, base_speed=set_speed)
        controller.set_desired(adaptive_speed)

        # 更新油门
        throttle = controller.update(speed, steering_angle)

        # 发送控制命令
        send_control(steering_angle, throttle)

        # 输出调试信息
        if frame_count % 10 == 0:  # 每10帧输出一次
            print(f"时间: {elapsed_time:.1f}s | 速度: {speed:.1f} mph | "
                  f"转向角: {steering_angle:.3f} | 油门: {throttle:.3f} | "
                  f"目标速度: {adaptive_speed:.1f} mph")

        frame_count += 1
    else:
        sio.emit('manual', data={})


@sio.on('disconnect')
def on_disconnect(sid):
    """断开连接回调"""
    print('与模拟器断开连接')


# 10、启动运行
if __name__ == '__main__':
    print('自动驾驶系统启动中...')
    print(f'基础速度: {set_speed} mph')
    print('等待模拟器连接...')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)