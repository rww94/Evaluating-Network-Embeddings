'''
NE-evaluation-link-pred
Date:2018-05-25
authot:raowei
'''
import numpy as np
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score

#Nodes数组，存储所有的节点ID
Nodes=[]
#Edges数组，存储所有边信息 第一个数源节点 第二个数目标节点
Edges=[]
#网络向量
Embeddings=[]
#X_test=[]测试数据集，第一个数源节点 第二个数目标节点 第三label
X_test=[]
#Node_map节点映射函数
Node_map={}
#节点邻居列表
Node_neighbors_map={}
#加载节点数据
def load_node(nodefile_):
    nodes=[]
    with open(nodefile_) as f:
        for l in f:
            nodes.append(int(l))
    return nodes
#加载边数据
def load_edge(edgefile_):
    edges=[]
    with open(edgefile_) as f:
        for l in f:
            x,y=l.strip().split()[:2]
            edges.append([int(x),int(y)])
    return edges
#加载预测数据
def load_test(testfile_,node_map):
    x_test = []
    with open(testfile_) as f:
        for l in f:
            x,y,z=l.strip().split()[:3]
            x_test.append([node_map[int(x)],node_map[int(y)],int(z)])
    return x_test
#加载向量数据
def load_embeddings(embeddingfile_):
    embeddings=np.loadtxt(embeddingfile_)
    return embeddings

#生成节点的映射数组和节点的邻居数组集
def make_map(nodes,edges):
    node_neighbors_map={}
    i=0
    node_map={}
    for node in nodes:
        node_map[node]=i
        i=i+1
    for edge in edges:
        if node_map[edge[0]] not in node_neighbors_map:
            node_neighbors_map[node_map[edge[0]]]=set([node_map[edge[1]]])
        else:
            node_neighbors_map[node_map[edge[0]]].add(node_map[edge[1]])
    return node_map,node_neighbors_map

#距离计算公式
def calculate_distance(embeddings, type): # N * emb_size
    if type == 'euclidean_distances':
        Y_predict = 1.0 * euclidean_distances(embeddings, embeddings)
    if type == 'cosine_similarity':
        Y_predict = cosine_similarity(embeddings, embeddings)
    return Y_predict

def norm(a):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * a[i]
    return math.sqrt(sum)

#余弦相似度
def cosine_similarity( a,  b):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * b[i]
    return sum/(norm(a) * norm(b))

#ROC指标
def evaluate_ROC(X_test, Embeddings):
    y_true = [ X_test[i][2] for i in range(len(X_test))]
    y_predict = [ cosine_similarity(Embeddings[X_test[i][0],:], Embeddings[X_test[i][1], :]) for i in range(len(X_test))]
    roc = roc_auc_score(y_true, y_predict)
    if roc < 0.5:
        roc = 1 - roc
    return roc

#平均准确率
def evaluate_MAP( node_neighbors_map, Embeddings, distance_measure):
    MAP = .0
    Y_true = np.zeros((len(node_neighbors_map), len(node_neighbors_map)))
    for node in node_neighbors_map:
        # prepare the y_true
        for neighbor in node_neighbors_map[node]:
            Y_true[node][neighbor] = 1
    print(distance_measure)
    Y_predict = calculate_distance(Embeddings,distance_measure)
    for node in node_neighbors_map:
        MAP +=  average_precision_score(Y_true[node,:], Y_predict[node,:])

    return MAP/len(node_neighbors_map)

if __name__=='__main__':
    '''
    inputfile:节点编号文件、边集文件edgelist类型、向量文件（不包括节点ID）与节点编号文件对应、链路预测文件
    output：ROC值，节点小于10000可以使用evaluate_MAP（不用链路预测文件）
    '''
    Nodes = load_node('nodes.txt')
    Edges = load_edge('doublelink.edgelist')
    Node_map, Node_neighbors_map = make_map(Nodes, Edges)
    X_test = load_test('test_pairs.txt',Node_map)
    Embeddings = load_embeddings('embs.txt')
    ROC=evaluate_ROC(X_test,Embeddings)
    print(ROC)
    #MAP=evaluate_MAP(Node_neighbors_map[:1000,:],Embeddings[:1000,:],'cosine_similarity')
    #print(MAP)
