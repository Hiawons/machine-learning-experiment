import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def gen_clusters(size1, size2):
    '''
    产生两类线性可分数据集
    :param size1: 数据集1的规模
    :param size2: 数据集2的规模
    :return labeldata: 带标签的总数据集
    '''
    labeldata = np.zeros((size1+size2, 3))
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean1, cov1, size1)
    labeldata[0:size1, 0:2] += data
    labeldata[0:size1, 2] = -1

    mean2 = [5, 5]
    cov2 = [[1, 0], [0, 1]]
    data = np.random.multivariate_normal(mean2, cov2, size2)
    labeldata[size1:size1+size2, 0:2] += data
    labeldata[size1:size1+size2, 2] = 1

    return np.round(labeldata,3)


def save_data(data, filename):
    with open(filename, 'w') as file:
        for i in range(data.shape[0]):
            file.write(str(data[i, 0]) + ',' + str(data[i, 1]) + ',' + str(data[i, 2]) + '\n')


def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            data.append([float(i) for i in line.split(',')])
    return np.array(data)



def show_scatter(data):
    '''
    :param data: 数据集
    :return:
    在画布上绘制散点图
    '''
    plt.ioff()
    x, y, label = data.T
    plt.scatter(x, y, c = label)
    plt.axis([-4, 9, -4, 9])
    plt.xlabel("x")
    plt.ylabel("y")

def draw_line(a, k):
    '''
    :param a: 直线参数向量
    :param k: 迭代次数
    在画布上绘制第k次迭代后的直线
    :return:
    '''
    global lines, fig, text
    fig = plt.gcf()
    x1, y1 = -3, ((-a[0, 2] - a[0, 0] * (-3))) / a[0, 1]
    x2, y2 = 8, ((-a[0, 2] - a[0, 0] * (8))) / a[0, 1]
    try:
        fig.gca().lines.remove(lines[0])  # 抹除
        plt.setp(text, visible=False)
    except Exception:
        pass
    lines = fig.gca().plot([x1, x2], [y1, y2], 'r-', lw=5) # 线的形式

    text = plt.text(2, 9, 'iteration k = %s' % k, ha='center', va='bottom', fontsize=12, visible=True)



def Fixed_incremenet_Single_sample_train(labeled_training_set):
    '''
    固定学习率单点训练算法，对于线性可分的数据集，一定会收敛。
    训练的迭代过程调用画线函数进行画图展示
    :param labeled_training_set: 带标签线性可分数据集
    :return plt.gcf()：画布
             aes:  包含每次迭代产生的直线参数a的列表
    '''
    aes = list()
    show_scatter(labeled_training_set)
    x1, x2, t = labeled_training_set.T
    y = np.ones((3, labeled_training_set.shape[0]))
    y[0, :] = x1
    y[1, :] = x2

    y = np.multiply(t, y)
    a = np.ones((1, 3))
    b = a.copy()
    aes.append(b)
    k = 0
    draw_line(a, k)
    finish = False
    while finish == False:
        finish = True
        task = np.dot(a, y)
        for i in range(labeled_training_set.shape[0]):
            if task[0, i] < 0:
                finish = False
                a += y[:, i]
                b = a.copy()
                aes.append(b)
                k = k + 1
                draw_line(a, k)
                plt.pause(0.5)
                task = np.dot(a, y)
    plt.savefig("result.png")
    plt.show()
    return plt.gcf(), aes

def update(frame):
    a = aes[frame]
    draw_line(a, frame)

def init():
    show_scatter(d)





#data = gen_clusters(100,150)
#save_data(data, '3clusters.txt')
d = load_data('3clusters.txt')
fig, aes = Fixed_incremenet_Single_sample_train(d)
anim = FuncAnimation(fig, update, frames=len(aes), init_func=init, interval=500)
anim.save('learn.gif', writer='imagemagick')