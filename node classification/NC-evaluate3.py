import tensorflow as tf
from numpy.random import RandomState
import random
import numpy as np


class Network(object):
    def __init__(self,sizes,random_seed=2018):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.random_seed = random_seed
        self._init_graph()

    #初始化计算图
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.train_x = tf.placeholder(tf.float32,shape=(None,64))
            self.train_y = tf.placeholder(tf.float32,shape=(None,10))
            self.default_initialier()
            #前向传播算法
            temp = self.train_x
            for b, w in zip(self.biases, self.weights):
                _y = tf.sigmoid(tf.matmul(temp,w)+b)
                temp = _y
            #交叉熵损失函数    
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.train_y,logits=_y))
            self.optimizer = tf.train.AdadeltaOptimizer(3.0).minimize(self.loss)
            init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(init)


    def default_initialier(self):
        self.weights = [tf.Variable(tf.random_normal([x,y]))
                      for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        self.biases = [tf.Variable(tf.zeros([1,y]))
                      for y in self.sizes[1:]]

    #批处理
    def update_mini_batch(self,mini_batch):
        x,y = zip(*mini_batch)
        x=np.reshape(x, (len(mini_batch), 64))
        y=np.reshape(y, (len(mini_batch), 10))
        feed_dict = {self.train_x:list(x),self.train_y:list(y)}
        opt,loss = self.sess.run((self.optimizer,self.loss),feed_dict=feed_dict)
        return loss

    #训练函数
    def train(self,train_data,epochs=20,batch_size=10,test_data=None):
        n=len(train_data)
        for epoch in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k + batch_size]for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch)
            print 'loss:{}'.format(loss)
            #性能评价
            if test_data:
                total = len(test_data)
                x, y = zip(*test_data)
                test_x = np.reshape(x, (total, 64))
                test_y = np.reshape(y, (total, 10))
                with self.graph.as_default():
                    self.test_x = tf.placeholder(tf.float32, shape=(None, 64))
                    self.test_y = tf.placeholder(tf.float32, shape=(None, 10))
                    test_dict = {self.test_x:test_x,self.test_y:test_y}
                    temp=self.test_x
                    for b, w in zip(self.biases, self.weights):
                        _y = tf.sigmoid(tf.matmul(temp, w) + b)
                        temp = _y
                    correct_prediction = tf.equal(tf.argmax(_y,1),tf.argmax(test_y,1))
                    accuracy = self.sess.run(tf.reduce_sum(tf.cast(correct_prediction,tf.float32)),
                                             feed_dict = test_dict)
                    print 'accuracy:{}\n'.format(accuracy/total)

from sklearn.cross_validation import train_test_split
def make_map(nodes):
    node_neighbors_map={}
    i=0
    node_map={}
    for node in nodes:
        node_map[node]=i
        i=i+1
    return node_map

def vectorized_result(j):
    e = np.zeros((1, 10))
    e[0][j] = 1.0
    return e

def split_data(labelfile_,nodefile_,embeddingfile_,len):
    train_size = 0.8
    random_state = 4
    with open(labelfile_) as f:
        label_data = []
        for l in f:
            line = l.strip().split()
            label_data.append([int(line[0]), int(line[1])])
    label_data = np.array(label_data)
    train, test = train_test_split(label_data, train_size=train_size, random_state=random_state)
    train_y=train[:,1]
    test_y=test[:,1]
    nodes=[]
    with open(nodefile_) as f:
        for l in f:
            nodes.append(int(l))
    node_map=make_map(nodes)
    embeddings=np.loadtxt(embeddingfile_)
    train_vec=[embeddings[node_map[i]] for i in train[:,0]]
    test_vec=[embeddings[node_map[i]] for i in test[:,0]]
    train_inputs = [np.reshape(x, (1, len)) for x in train_vec]
    train_results = [vectorized_result(y) for y in train_y]
    train_data = list(zip(train_inputs, train_results))
    test_inputs = [np.reshape(x, (1, len)) for x in test_vec]
    test_results = [vectorized_result(y) for y in test_y]
    test_data = list(zip(test_inputs, test_results))
    return train_data,test_data


if __name__=='__main__':

    labelfile_='labels.txt'
    nodefile_='nodes.txt'
    embeddingfile_='embs.txt'
    #向量的维度
    embs_dimension = 64
    train_data,test_data=split_data(labelfile_,nodefile_,embeddingfile_,embs_dimension)
    #网络每层的大小
    net=Network([64,60,20,10])
    net.train(train_data,epochs=20,batch_size=10,test_data=test_data)
