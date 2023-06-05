from __future__ import print_function
import numpy as np
import os
import logging
# 欧氏距离
def eucliDist(img1,img2,isFc=False):
    if isFc:
        return (1.0/(1.0+np.linalg.norm(img1 - img2)))
    return (1.0/(1.0+np.linalg.norm(img1 - img2,axis=(1,2)))).mean()

# 余弦相似度
def Cosine(img1,img2):
    a = img1
    b = img2
    dist = np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b))
    return dist

def channles_selected(FM,t=1,s=0.0):
    U = [i for i in range(FM.shape[1])]
    K = []
    R = []
    for i in range(len(U)):
        if U[i] is None:continue
        K.append(i)
        scores = []
        img1 = FM[:,i,:,:].detach().numpy()
        for j in range(len(U)):
            if U[j] is None or i == j: 
                scores.append((j,0))
                continue
            img2 = FM[:,j,:,:].detach().numpy()
            score = eucliDist(img1,img2)
            scores.append((j,score))
        scores.sort(key=lambda x:x[1],reverse=True)
        for index in range(0,t):
            s_ = scores[index][1]
            i_ = scores[index][0]

            if i_ not in K and s_ >= s:
                R.append(i_)
                U[i_] = None
    return K



def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging
