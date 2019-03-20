
# coding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import time
# 创建保存模型和结果的文件夹
result_path = './result'
model_path = './result/model'
if not os.path.exists(model_path):
    os.makedirs(model_path)#存放模型
if not os.path.exists(result_path):
    os.makedirs(result_path)#存放日志和结果
# 创建日志文件
if not os.path.exists('./result/date.txt'):
    with open('./result/date.txt', 'w') as acc_file:
        pass
with open('./result/date.txt', 'a') as acc_file:
    acc_file.write('\n%s \n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))

#数据导入
data = pd.read_csv("data/data.csv",)
x = data.iloc[:,:13]
y = data.MEDV
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

#此处为将数据变为正态分布，可变也可不变
# X_train = scale(X_train)
# X_test = scale(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# y_train = scale(y_train.reshape((-1,1)))
# y_test = scale(y_test.reshape((-1,1)))

#NETWORK TOPOLOGIES设置模型框架的参数
n_hidden_1 = 10 
n_hidden_2 = 5 
n_hidden_3 = 3 
n_input    = 13 
n_classes  = 1 

xs = tf.placeholder(shape=[None,X_train.shape[1]],dtype=tf.float32)
ys = tf.placeholder(shape=[None,1],dtype=tf.float32)
keep_prob_s = tf.placeholder(dtype=tf.float32)
pos = tf.placeholder(dtype=tf.float32)

#设置初始的权重和偏置项
stddev = 0.1#标准差
weights = {
    'W1': tf.Variable(tf.random_normal([n_input,n_hidden_1], stddev=stddev)),
    'W2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2], stddev=stddev)),
    'W3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3], stddev=stddev)),    
    'out': tf.Variable(tf.random_normal([n_hidden_3,n_classes],stddev=stddev))
}

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.random_normal([n_hidden_3])),    
    'out' : tf.Variable(tf.random_normal([n_classes]))
}


#搭建神经网络(我这里分别试了一下2层、3层、4层的神经网络，并且发现随着层数的增加，非线性越来越好，对应的拟合效果也越好，但是层数肯定有极限！)
def multilayer_perceptron(_X, _weight, _biases,_keepratio):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weight['W1']), _biases['b1']))
    dropout_1 = tf.nn.dropout(layer_1,keep_prob=_keepratio)
    
    layer_2 = tf.nn.relu(tf.add(tf.matmul(dropout_1,_weight['W2']),_biases['b2']))
    dropout_2 = tf.nn.dropout(layer_2,keep_prob=_keepratio)
    
    layer_3 = tf.matmul(dropout_2,_weight['W3']) + _biases['b3']
    dropout_3 = tf.nn.dropout(layer_3,keep_prob=_keepratio)
    
    layer_4 = tf.matmul(dropout_3,_weight['out']) + _biases['out']
    dropout_4 = tf.nn.dropout(layer_4,keep_prob=_keepratio)    
    
    return dropout_4
    
#设置超参数
keep_prob=1  # 防止过拟合，取值一般在0.5到0.8。我这里是1，没有做过拟合处理
ITER =10000  # 一共迭代1万次，考虑样本数据比较少
p = 0.1
#预测
pred = multilayer_perceptron(xs,weights,biases,keep_prob_s)
#定义损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_train - pred),reduction_indices=[1])) 
#优化器（这里有两个优化器都可以选择，可以比较结果那个好）
optm = tf.train.AdamOptimizer(learning_rate = pos).minimize(loss)
# optm = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss) 

#没迭代100次保存一个最优的模型
save_step = 100
saver = tf.train.Saver(max_to_keep=1) #最多保存1个模型
one = tf.constant(0.1,dtype=tf.float32)#学习率的迭代
def fit(X,y,n,keep_prob,p):
    init = tf.global_variables_initializer()  
    stage_epochs = [5000,3000,2000]#分批次进行训练
    with tf.Session() as sess:
        tf.global_variables()
        sess.run(init)
        end = time.time()
        loss1 = 50#初始的损失值设为50，一般的回归问题，初始loss是无穷大的
        for i in range(n):
            if (i + 1) in np.cumsum(stage_epochs)[:-1]:
                p = sess.run(tf.multiply(p,one))
                print('  ---Step into next stage---learning_rate: %.5f'%p)
                with open('./result/date.txt', 'a') as acc_file:
                        acc_file.write('-----Step into next stage---learning_rate: %.5f----\n'%p)
                        
                feed_dict_train = {ys: y, xs: X, keep_prob_s : keep_prob,pos : p}
                _loss,_ = sess.run([loss,optm], feed_dict = feed_dict_train)  
            else:
                feed_dict_train = {ys: y, xs: X, keep_prob_s : keep_prob,pos : p}
                _loss,_ = sess.run([loss,optm], feed_dict = feed_dict_train)
            #每100次进行打印一次结果。
            if i%200 == 0:
                print('epoch:%d\t loss:%.8f\t Time:%.5f\t' % (i,_loss,time.time()-end))
                end = time.time()
                with open('./result/date.txt', 'a') as acc_file:
                    acc_file.write('epoch:%d\t loss:%.8f\t\n' % (i,_loss))
            #保存模型
            if i%save_step == 0:
                if _loss < loss1:      
                    saver.save(sess, "result/model/model.ckpt")    
                    loss1 = _loss


fit(X=X_train,y=y_train,n=ITER,keep_prob = keep_prob,p = p)


#打开保存的最优model并预测测试集数据
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./result/model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint("result/model/"))
    y_pred = sess.run(pred, feed_dict={ys: y_test, xs: X_test, keep_prob_s : keep_prob,pos : p})
    print("Test set predicted complete !/n")
    
#可视化比较测试集与预测的差别
print("Visually demonstrate test and predicted values")
plt.plot(range(len(y_test)),y_test,'b')
plt.plot(range(len(y_pred)),y_pred,'r--')
plt.legend(['y_test','y_pred'])
plt.title("Test set and predicted values")
plt.savefig('./result/result%s.jpg'%time.strftime('%H-%M-%S',time.localtime(time.time())))
plt.show()
