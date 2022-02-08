from multiprocessing import Queue
import threading
import paho.mqtt.client as mqtt 
import json
import ctypes
import inspect
import sys
from multiprocessing import Process
import psutil



# from app import PID_TASK
sys.path.append("..")
sys.path.append("...")
sys.path.append("/data0/BigPlatform/ZJPlatform/015_MI/CopycatTest/")
import os
print("import all_cam_dj")

""" 常量定义 """ 
# EMQ 订阅topic

# 定义第一步，定义TOPIC
TOPIC_STOP_TASK     = 'stopTask'
TOPIC_ATTACK_START  = 'startTask'

TOPIC_ATTACK_PAUSE = "pauseTask"
TOPIC_ATTACK_CONSUME = "continueTask"
# 最大线程数/任务数量
MAX_THREAD = 5
# 任务字典，根据任务id索引任务执行的Future
taskMap = {}
global FLAG
# global TaskPID

""" 路由方法 """
# 停止任务
def pause_task(queue: Queue, topic, payload):

    # 解析data里面的数据
    dataModal = payload["dataModal"]  # 检测所用的数据模态
    taskType = payload["taskType"]
    if dataModal != "IMAGE":
        return None
    if taskType == "Federal":
        return None
    print(payload)
    print(TaskPID)
    p = psutil.Process(TaskPID)
    p.suspend()  # 挂起进程


def continue_task(queue: Queue, topic, payload):

    # 解析data里面的数据
    dataModal = payload["dataModal"]  # 检测所用的数据模态
    taskType = payload["taskType"]
    if dataModal != "IMAGE":
        return None
    if taskType == "Federal":
        return None
    print(payload)
    p = psutil.Process(TaskPID)
    p.resume()  # 挂起进程





# 停止任务
def stop_task(queue: Queue, topic, payload):
    # print(payload)
    # 解析data里面的数据
    dataModal = payload["dataModal"]  # 检测所用的数据模态
    taskType = payload["taskType"]
    if dataModal != "IMAGE":
        return None
    if taskType == "Federal":
        return None
    print(payload)
    p = psutil.Process(TaskPID)
    p.kill()  # 杀死进程
    # taskId = payload['taskId']
    # taskset = taskMap[str(taskId)]
    # print(type(taskset))
    # global FLAG
    # FLAG = 1
    # if type(taskset) is not list:
    #     new = []
    #     new.append(taskset)
    #     taskset = new
    # for task in taskset:
    #     if task.is_alive():
    #         stop_thread(task)

def attack_start_process(queue: Queue, topic, payload):

    data = payload["data"]
    # 解析data里面的数据
    dataModal = data["dataModal"]  # 检测所用的数据模态
    if dataModal != "IMAGE":
        return None

    taskType = payload["taskType"]

    if taskType == "Federal":
        return None
    print(payload)
    global TaskPID
    TaskSUB = Process(target=attack_start,args=(queue, topic, payload))

    # TaskSUB.daemon = True
    TaskSUB.start()
    print("TaskPID", TaskSUB.is_alive())
    TaskPID = TaskSUB.pid
    print("TaskPID", TaskPID)





def MultiTherad(ParamsSet,target):
    """

    :param ParamsSet: 输入的参数集合
    :param target: 需要调用的函数名
    :return:
    """
    Thread = []
    global FLAG
    FLAG = 0
    for i in range(len(ParamsSet)):
        Thread.append(threading.Thread(target=target, args=(ParamsSet[i],)))
    for i in range(len(ParamsSet)):
        taskMap[str(ParamsSet[i]["taskId"])] = Thread
    for i in range(0,len(ParamsSet),2):
        print(i,len(ParamsSet))
        if i+1>=len(ParamsSet):
            Thread[i].setDaemon(True)
            Thread[i].start()
            Thread[i].join()
        else:
            Thread[i].setDaemon(True)
            Thread[i+1].setDaemon(True)
            Thread[i].start()
            Thread[i+1].start()
            Thread[i].join()
            Thread[i+1].join()
            if FLAG == 1:
                break


def SingleTherad(ParamsSet,target):
    print("单线程")
    Thread = []
    global FLAG
    FLAG = 0
    for i in range(len(ParamsSet)):
        Thread.append(threading.Thread(target=target, args=(ParamsSet[i],)))
    for i in range(len(ParamsSet)):
        taskMap[str(ParamsSet[i]["taskId"])] = Thread
    for i in range(0,len(ParamsSet),1):
        Thread[i].setDaemon(True)
        Thread[i].start()
        Thread[i].join()
        if FLAG == 1:
            break



# 第二步，开始测试任务
def attack_start(queue: Queue, topic, payload):


    taskId = payload['taskId'] #用户ID，用来构建唯一存放路径时使用
    taskType = payload["taskType"]#检测是单机模式还是联邦学习模式
    data = payload["data"]

    #解析data里面的数据
    dataModal = data["dataModal"]  #检测所用的数据模态
    dataEvaluationObject = data["evaluationObject"]
    dataSceneName = data["sceneName"]
    dataPlatform = data['platformFrame']
    dataDataset = data['dataset']
    dataDepthModel = data['depthModel']

    methods = data["methods"]
    step = methods[0]["step"]
    method = methods[0]["attackMethod"] #########

    if dataModal != "IMAGE":
        return None

    # 执行算法部分
    # TODO
    # 首先检查任务队列是否已经满
    if can_doing():
        # TODO START

        # try:
        # 重新包装攻击参数集合


        # 把攻击参数包装拆开后组成N个字典，准备进行多线程
        attackParamsSet = []

        # Step1：从payload解析出需要的参数集合
        for i in range(len(methods)):
            attackParamsSet.append(
                {
                    'platform': dataPlatform,
                    'model': dataDepthModel,
                    'dataset': dataDataset,
                    'taskId': taskId,
                    'queue': queue,
                    'method': method,
                }
            )
        if step=='STEAL_DETECTION':
            if method=='LDVM':
                from LDVM import LDVM
                t = threading.Thread(target=SingleTherad, args=(attackParamsSet, LDVM,))
            elif method=='TCDVM':
                from TCDVM import TCDVM
                t = threading.Thread(target=SingleTherad, args=(attackParamsSet, TCDVM,))
            t.start()
            t.join()
            print("成功开启")
        if step=="STEAL_ASSESSMENT":
            if dataEvaluationObject == "BuiltInSystem":
                # Step2:载入需要用到的函数
                if method=='Steal':
                    from Steal import main
                    t = threading.Thread(target=SingleTherad, args=(attackParamsSet, main,))
                elif method=='copycat':
                    from copycat import copycat
                    t = threading.Thread(target=SingleTherad, args=(attackParamsSet, copycat,))
                # Step3：定义多线程工作
                # MultiTherad(attackParamsSet,Image_Attack)
                
                # t.setDaemon(True)
                t.start()
                t.join()
                print("成功开启")


            if dataEvaluationObject == "CommercialAPI":
                # Step2:载入需要用到的函数
                if dataDataset!="NSFW-API":
                    from Image_Attack_API_DJ import Image_Attack
                if dataDataset=="NSFW-API":
                    from Image_Attack_API_JIanHuang_DJ import Image_Attack
                # Step3：定义多线程工作
                print("商用")


                t = threading.Thread(target=SingleTherad,
                                     args=(attackParamsSet, Image_Attack,))
                t.start()
                t.join()

                taskMap[str(taskId)] = t


        # Step2：从payload解析出需要的参数
        NeuralParamsSet = attackParamsSet
        if step=="NEURON_VIS":

            # Step2:载入需要用到的函数
            from Image_Testing import Image_Testing
            # Step3：定义单线程工作

            t = threading.Thread(target=SingleTherad,
                                 args=(NeuralParamsSet, Image_Testing,))
            # t.setDaemon(True)
            t.start()
            t.join()

            taskMap[str(taskId)] = t

        # Step1：从payload解析出需要的参数
        VisParamsSet = attackParamsSet
        if step =="SAMPLE_VIS":
            # Step2:载入需要用到的函数
            from all_cam_dj2 import Main as ImageVis
            # Step3：定义多线程工作
            t = threading.Thread(target=MultiTherad,
                                 args=(VisParamsSet, ImageVis,))
            # t.setDaemon(True)
            t.start()
            t.join()

            taskMap[str(taskId)] = t

        # Step1：从payload解析出需要的参数
        TSNEParamsSet = attackParamsSet
        if step =="EDGE_VIS":
            # Step2:载入需要用到的函数
            from tsne import TSNE
            # Step3：定义多线程工作
            # MultiTherad(TSNEParamsSet, TSNE)
            t = threading.Thread(target=SingleTherad,
                                 args=(TSNEParamsSet, TSNE,))
            # t.setDaemon(True)
            t.start()
            t.join()

            taskMap[str(taskId)] = t

        # Step1：从payload解析出需要的参数
        RobustParamsSet = attackParamsSet
        if step == "ROBUST":
            # Step2:载入需要用到的函数
            from Image_Robustness import Image_Robustness
            # Step3：定义多线程工作
            # MultiTherad(RobustParamsSet, Image_Robustness)
            t = threading.Thread(target=SingleTherad,
                                 args=(RobustParamsSet, Image_Robustness,))
            # t.setDaemon(True)
            t.start()
            t.join()

            taskMap[str(taskId)] = t

        # except:
        #     print("多线程有问题的，淦")




        # TODO END

    else:
        complete = {
            'topic': 'finish',
            'data': {
                'taskId': taskId,
                'message': '任务队列已满'
            }
        }
        queue.put(complete)
    # global FLAG
    # while not FLAG:
    #     pass
    #
    # taskId = payload['taskId']
    # taskset = taskMap[str(taskId)]
    # if type(taskset) is not list:
    #     new = []
    #     new.append(taskset)
    #     taskset = new
    # for task in taskset:
    #     if task.is_alive():
    #         stop_thread(task)
    # print("完成")


def no_topic(queue: Queue, topic, payload):
    """
        不支持的算法类型
    """
    taskId = payload['taskId']
    # payload是创建攻击的参数
    complete = {
        'topic' : 'finish',
        'data' : {
            'taskId' : taskId,
            'message' : '不支持此类型的计算:{}'.format(topic)
        }
    }
    queue.put(complete)

def consume(client: mqtt.Client, queue: Queue):
    """ 这是一个队列任务消费者

        守护线程方法，用于从队列中获取数据并发送给指定topic
    """
    while True:
        obj = queue.get()
        topic = obj['topic']
        data = obj

        # 发送消息
        client.publish(topic, json.dumps(data, ensure_ascii=False), 1)
        # queue.task_done()


def can_doing() -> bool:
    """ 检查能否执行一个新的任务

        计算当前正在执行的线程数量，如果大于等于5个则返回False。
        如果有部分线程已经执行完毕，则将其从taskMap中删除
    """
    # 线程结束队列
    complete_list = []
    # 活动线程数
    alive_counter = 0



    for key, value in taskMap.items():
        if value is list:
            if value[0].is_alive() == False:
                complete_list.append(key)
            else:
                alive_counter += 1


    for taskId in complete_list:
        taskMap.pop(taskId)
    return alive_counter < 5

def __async__raise(tid, exctype):
    """ 通过引发异常结束线程 """
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread: threading.Thread):
    """
        线束线程
    """
    __async__raise(thread.ident, SystemExit)

# 服务类
class Server(object):
    def __init__(self, client: mqtt.Client):
        self.client = client
        self.router = {
            TOPIC_STOP_TASK : stop_task,
            TOPIC_ATTACK_START : attack_start_process,
            TOPIC_ATTACK_PAUSE : pause_task,
            TOPIC_ATTACK_CONSUME: continue_task,
        }
        self.queue = Queue(maxsize=10)  # 最大允许10个任务堆积

    def register(self):
        """
            给emq客户端注册对应的topic
        """
        self.client.subscribe(TOPIC_STOP_TASK)           # 终止任务
        self.client.subscribe(TOPIC_ATTACK_START)        # 开始攻击任务
        self.client.subscribe(TOPIC_ATTACK_PAUSE)  # 暂停攻击任务
        self.client.subscribe(TOPIC_ATTACK_CONSUME)  # 重启攻击任务

        t = threading.Thread(target=consume, args=(self.client, self.queue))
        t.setDaemon(True)
        t.start()



    def do_message(self, topic, payload):
        # global TaskPID
        """
            执行对应topic的任务
        """

        # print(payload)
        # print("_++++++++++++++++++++++++{}".format(topic))
        # TaskPID = Process(target=self.router.get(topic, no_topic),args=(self.queue, topic, payload))
        # # TaskPID.daemon = True
        # TaskPID.start()
        # TaskPID.join()

        self.router.get(topic, no_topic)(self.queue, topic, payload)

