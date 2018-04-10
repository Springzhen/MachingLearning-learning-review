# -*- coding: UTF-8 -*-

#kNN: k Nearest Neighbors

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

from numpy import *
import operator
from os import listdir

#k-NN算法
def classify0(inX, dataSet, labels, k):
    '''
    输入：inX 用于分类的输入向量--testdata
      dataSet  输入的训练样本集--traindata
      labels  标签向量--train label
      k  选择最近邻居的数目
    输出  最近邻的标签
    '''
    dataSetSize = dataSet.shape[0]    #返回行数
    #print 'tile',tile(inX, (dataSetSize, 1))
    #print 'dataSetSize',dataSetSize
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #将inX转化为 dataSetSize行1列 instance:arr = np.array([[1,2,3],[4,5,6]])
    #print 'diffMat',diffMat                                                             #brr = np.array([[1,1,3],[2,5,6]])
                                                                                        #crr = brr-arr
                                                                                        #    =np.array([[0,-1,0],[-2,0,0]])
    sqDiffMat = diffMat**2    #求平方
    sqDistances = sqDiffMat.sum(axis=1)    #行相加
    distances = sqDistances**0.5   #开根号
    sortedDistIndicies = distances.argsort()    #从小到大排序，返回索引
    classCount={}# get class count
    for i in range(k):#
        voteIlabel = labels[sortedDistIndicies[i]]# select K-num tags
        print 'voteIlabel ',voteIlabel,classCount.get(voteIlabel,0)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #dict sort
    print 'sortedClassCount',sortedClassCount
    return sortedClassCount[0][0]

#创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#将文本转换为Numpy数据
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #获取文档的行数
    returnMat = zeros((numberOfLines,3))        #文档行数*3的0向量矩阵
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]   #前3列作为特征存储
        classLabelVector.append(int(listFromLine[-1]))   #将标签存储
        index += 1
    return returnMat,classLabelVector

#归一化特征值    
def autoNorm(dataSet):
    minVals = dataSet.min(0)   #沿行向量的最小值（列中选取最小的值）
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #print 'minVals',minVals
    #print 'maxVals',maxVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]# row num
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #归一化结果   nordata = (x - min)/(max - min)
    return normDataSet, ranges, minVals

#分类器测试代码    
def datingClassTest(k=3):
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]#raw nums
    numTestVecs = int(m*hoRatio) #100
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print "the total accuracy is: %f" % (1-errorCount/float(numTestVecs))
    print errorCount

#图片数据转换为行向量 reshape image from 32*32 to 1*1024
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename) #pic size: 32*32
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []#train data tag list
    trainingFileList = listdir('trainingDigits')           #load the training set
    #print 'trainingFileList',trainingFileList
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))# training image every file to 1*1024
    #print 'trainingMat ',trainingMat
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0]) #get class tag
        hwLabels.append(classNumStr)
        a = 'trainingDigits/%s' % fileNameStr
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0]) #test data class tags
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 5)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
if __name__=='__main__':
    #handwritingClassTest()
    for i in range(1,10): datingClassTest(k=i)# 选择最优临近数目k
