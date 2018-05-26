'''
NE-evaluation-node-class
Date:2018-05-26
authot:raowei
'''
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import time

def split_data(labelfile_,nodefile_,embeddingfile_):
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
    return train_vec,test_vec,train_y,test_y

def make_map(nodes):
    node_neighbors_map={}
    i=0
    node_map={}
    for node in nodes:
        node_map[node]=i
        i=i+1
    return node_map

def evaluation(labelfile_, nodefile_, embeddingfile_,classifierStr='SVM'):
    time_start=time.time()
    train_vec, test_vec, train_y, test_y=split_data(labelfile_, nodefile_, embeddingfile_)
    if classifierStr=='SVM':
        clf = SVC(kernel='linear')  # 线性分类
    elif classifierStr=='KNN':
        clf = KNeighborsClassifier()
    else:
        clf=MultinomialNB()
    clf.fit(train_vec,train_y)
    y_pred = clf.predict(test_vec)
    time_end = time.time()
    print('evaluation runtime:{}'.format(time_end-time_start))
    cm = confusion_matrix(test_y, y_pred)
    #print(cm)
    acc = accuracy_score(test_y, y_pred)
    print('Accuracy:{}'.format(acc))
    macro_f1 = f1_score(test_y, y_pred, pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred, pos_label=None, average='micro')
    #print(macro_f1,micro_f1)

if __name__=='__main__':
    labelfile_ = 'labels.txt'    #分类标签文件
    nodefile_ = 'nodes.txt'      #节点顺序文件与向量文件对应
    embeddingfile_ = 'embs.txt'  #向量文件
    evaluation(labelfile_, nodefile_, embeddingfile_, classifierStr='KNN')
