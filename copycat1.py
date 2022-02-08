import torch
import numpy as np
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms
from LoadData import dic,my_data_set
import argparse
import resnet
import torch.nn as nn
from tqdm import tqdm

    # api查询结果路径
apiFilePath = '/data0/BigPlatform/ZJPlatform/015_MI/net20/storage/'

    # 模型窃取训练数据路径
trainFilePath = testFilePath = '/data0/BigPlatform/ZJPlatform/015_MI/StealTest/StealData/'

    # 窃取模型存放路径
stealModelPath = '/data0/BigPlatform/ZJPlatform/015_MI/CopycatTest/Models/'

image_path='/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/'


def test():
    # global frame, taskid, queue, dataset, model, method, batch_size, apiFilePath, trainFilePath, testFilePath, stealModelPath, apiTrainName, apiTestName, trainFileName, testFileName, stealModelName
    imglist_fn = os.path.join(apiFilePath,testFileName)
    batch_size = 24
    data_num = correct = 0
    # max_epochs = max_epochs
    dict, cls_number = dic(imglist_fn,method)
    loss_function = torch.nn.CrossEntropyLoss()
    model = resnet.resnet18(pretrained=False, cls_number=cls_number)
    dataset = my_data_set(dict,method)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    print('Test model...')
    if os.path.exists(stealModelPath+stealModelName):
        print("加载模型继续训练...")
        model.load_state_dict(torch.load(stealModelPath+stealModelName))
    model = model.to(device)
    with tqdm(loader) as tqdm_loader:
        for i, (inputs, labels, credit) in enumerate(tqdm_loader):
            # print(inputs)
            data_num += labels.shape[0]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            print(correct / data_num)
    return correct / data_num

def copycat(params):
    # global frame, taskid, queue, dataset, model, method,batch_size, apiFilePath, trainFilePath, testFilePath, stealModelPath, apiTrainName, apiTestName, trainFileName, testFileName, stealModelName
    frame = params['platform']
    taskid = params['taskId']
    queue = params['queue']
    dataset = params["dataset"]
    model = params["model"]
    method = params["method"]
    print('框架：',frame)
    print('模型：',model)
    print('目标数据集：',dataset)
    print('窃取方法：',method)
    apiTrainName = frame + '_' + model + '_' + dataset.lower() + '_' + 'train.txt'
    apiTestName = frame + '_' + model + '_' + dataset.lower() + '_' + 'test.txt'
    trainFileName = frame + '_' + model + '_' + dataset.lower() + '_' + 'train.txt'
    testFileName = frame + '_' + model + '_' + dataset.lower() + '_' + 'test.txt'
    stealModelName = frame + '_' + model + '_' + dataset.lower() + '_' + method + '.pth'
    result = {
        'topic': 'taskData',  # 这个topic表示攻击结果的返回
        'taskId': taskid,  # 在这里携带taskId
        'step': "ATTACK",
        'data': {}  # 这里是“攻击结果返回”这个topic携带的对象
    }
    imglist_fn = os.path.join(trainFilePath,trainFileName)
    batch_size = 32
    # max_epochs = max_epochs
    dict,cls_number=dic(imglist_fn,method)
    loss_function = torch.nn.CrossEntropyLoss()
    model = resnet.resnet18(pretrained=False,cls_number=cls_number)
    dataset = my_data_set(dict,method)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    epoch=0
    print('Training model...')
    if os.path.exists(stealModelPath+stealModelName):
        print("加载模型继续训练...")
        model.load_state_dict(torch.load(stealModelPath+stealModelName))
    model = model.to(device)
    # cls = cls.to(device)
    while 1:
        epoch+=1
        model.train()
    # for epoch in range(max_epochs):
        data = {}
        running_loss = 0.0
        data_num = correct = 0
        with tqdm(loader) as tqdm_loader:
            for i, (inputs, labels,credit) in enumerate(tqdm_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                data_num += labels.shape[0]
                outputs = model(inputs)
                # print(outputs)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                # y_hat = stealModel(x)
                # test_acc_sum += (y_hat.argmax(dim=1) == y.squeeze(1)).sum().item()
                correct += (predicted == labels).sum().item()
                print(correct / data_num)
                # if i % 200 == 199:
                #     tqdm_loader.set_description('Epoch: {}/{} Loss: {:.3f}'.format(
                #         epoch+1, max_epochs, running_loss/200.))
                #     running_loss = 0.0
        torch.save(model.state_dict(), stealModelPath+stealModelName)
        acc=test()
        jsontext = {"Epoch": int(epoch), "AttackMethod": str(method), "AttackSuccessRate": float(acc)}
        result['data'] = jsontext
        # 结果put进这个队列中就好
        # 会有一个线程一直迭代队列然后将结果发给后端
        queue.put(result)  # 导致速度变慢
        time.sleep(1)
        if acc > 0.8:
            print('模型窃取结束...')
            complete = {
                'topic': 'taskFinish',
                'taskId': taskid,
                'step': 'ATTACK',
                "attackMethod": method, }
            # TODO END
            # 输出只需要将结果put进队列中
            queue.put(complete)
            break
    print('Model trained.')
    print('Saving the model to "{}"'.format(model_fn))
