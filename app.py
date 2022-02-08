import paho.mqtt.client as mqtt
import json
from myserver import Server
from multiprocessing import Process
# MQTT配置
MQTT_HOST = '100.100.20.206'
MQTT_PORT = 1883

# mqtt配置
mqtt_client = mqtt.Client()

# 创建服务对象
server = Server(mqtt_client)

# mqtt连接与接收终止任务
def on_connect(client, userdata, flags, rc):
    print('Connected with result code '+str(rc))
    if rc != 0:
        print('连接失败')
        exit(0)
    # 注册订阅信息
    server.register()

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = json.loads(msg.payload.decode('utf-8'))
    server.do_message(topic, payload)

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# 启动emqtt
mqtt_client.connect(MQTT_HOST, MQTT_PORT)

mqtt_client.loop_forever()

# PIDTask = Process(target=mqtt_client.loop_forever)
# PIDTask.daemon = True
# PIDTask.start()