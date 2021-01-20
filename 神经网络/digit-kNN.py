import csv
import operator

from numpy import *
from numpy import mat

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in range(m):
        for j in range(n):
                newArray[i,j]=int(array[i,j])
    return newArray

def nomalizing(array):
    m,n=shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def LoadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            # 把数据一行一行存入列表
            l.append(line) # 42001*785
    l.remove(l[0]) # 删除第一行文字描述 (now 42000*785)
    l = array(l)
    label = l[:,0]
    data = l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def LoadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0]) # 28001*784 -----> 28000*784
    data = array(l)
    return nomalizing(toInt(data))

def classify(inX, dataSet, labels, k):
    # 待分类的输入向量inX,输入的训练样本集dataset，标签向量labels, k表示用于选择最近邻居的数目
    # 调用numpy中mat()函数将输入向量inX转换为矩阵，然后可以对矩阵进行一些线性代数的操作
    inX = mat(inX)
    dataSet = mat(dataSet)
    labels = mat(labels)
    # 获取样本dataSet的行数
    dataSetSize = dataSet.shape[0]
    #先把标签向量重复dataSet行 1列生成一个和dataSetSize同维度的矩阵再减去dataSet
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = array(diffMat)**2
    #将矩阵每一行里面的所有元素相加形成一个dataSetSize行 1列的矩阵
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    #将所求距调用numpy中的argsort()函数离排序
    #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引号)
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlable = labels[0,sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable,0) + 1
    #函数 sorted 方法返回的是一个新的 list
    #  key=operator.itemgetter(1), reverse=True 以第二个值【itemgetter(1)】降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def saveResult(result):
    with open('result.csv','w',newline='') as file:
        filewriter = csv.writer(file)
        for i in result:
            tmp = []
            tmp.append(i)
            filewriter.writerow(tmp)

def classTest():
    trainData, trainLable = LoadTrainData()
    testData = LoadTestData()
    m, n = shape(testData)
    errorCount = 0
    resultList = []
    for i in range(m):
        classifierResult = classify(testData[i], trainData, trainLable, 5)
        resultList.append(classifierResult)
        print('the classifier came back with: %d' % classifierResult)
    saveResult(resultList)

classTest()
