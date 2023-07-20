# -*- coding: UTF-8 -*-

import threading
from time import sleep, ctime

import cv2
import os
import json

totalThread = 16  # 需要创建的线程数，可以控制线程的数量

config_file = '模型配置文件'
checkpoint_file = '模型权重文件'
test_data_dir = './result/result_label/'

listImg = [file for file in os.listdir(test_data_dir)]  #创建需要读取的列表，可以自行创建自己的列表
lenList = len(listImg)  #列表的总长度
gap = int(lenList / totalThread)  #列表分配到每个线程的执行数

# 按照分配的区间，读取列表内容，需要其他功能在这个方法里设置
def processSection(name, s, e):
    for i in range(s, e):
        processImg(name, listImg[i])


def processImg(name, filename):
    # 这个部分内容包括：
    # 1. 加载模型
    # 2. 根据file读取图片
    # 3. 将结果进行处理并进行保存
    print("Thread %s: have processed %s" % (name, filename))
    print(os.path.join('\t resultData', filename + '.json'), end="")



class myThread(threading.Thread):

    def __init__(self, threadID, name, s, e):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.s = s
        self.e = e

    def run(self):
        print("Starting " + self.name + ctime(), end="")
        print(" From %d to %d" % (self.s, self.e))
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        # 这里由于数据不存在冲突情况，所以可以注释掉锁的代码
        # threadLock.acquire()
        #线程需要执行的方法
        processSection(self.name, self.s, self.e)
        # 释放锁
        # threadLock.release()



threadLock = threading.Lock()  #锁
threads = []  #创建线程列表

# 创建新线程和添加线程到列表
for i in range(totalThread):
    thread = 'thread%s' % i
    if i == 0:
        thread = myThread(0, "Thread-%s" % i, 0, gap)
    elif totalThread == i + 1:
        thread = myThread(i, "Thread-%s" % i, i * gap, lenList)
    else:
        thread = myThread(i, "Thread-%s" % i, i * gap, (i + 1) * gap)
    threads.append(thread)  # 添加线程到列表

# 循环开启线程
for i in range(totalThread):
    threads[i].start()

# 等待所有线程完成
for t in threads:
    t.join()
print("Exiting Main Thread")
