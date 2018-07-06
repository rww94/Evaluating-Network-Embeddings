#title = {Towards Understanding the Geometry of Knowledge Graph Embeddings}
import numpy as np

def cosine_similarity(vec1,vec2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vec1,vec2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA**0.5) * (normB**0.5))

def Conicity(embs):
    l = len(embs)
    avg_vec = embs[0]
    for i in range(1,len(embs)):
        avg_vec += embs[i]
    avg_vec /= l
    Conic = 0.0
    ATMs = []
    for emb in embs:
        ATM = cosine_similarity(emb,avg_vec)
        Conic += ATM
        ATMs.append(ATM)
    Conic /= l
    return Conic,ATMs

def AVL(embs):
    l = len(embs)
    avl = 0.0
    for emb in embs:
        avl += np.linalg.norm(emb)
    avl /= l
    return avl

if __name__ == "__main__":
    _file = 'embs.txt'
    embs = np.loadtxt(_file)
    wf = open('analysis_result.txt','a')
    l = len(embs[0])
    con,atms = Conicity(embs)
    avl = AVL(embs)
    print 'embeddings dimensions:{}'.format(l)
    print 'Conicity:{}'.format(con)
    print 'AVL:{}'.format(avl)
    wf.write(str(l)+'----'+str(con)+'----'+str(avl)+'\n')
