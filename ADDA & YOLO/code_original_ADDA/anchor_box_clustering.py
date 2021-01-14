import glob
import numpy as np
from sklearn.cluster import KMeans
label_files = glob.glob('./data/labels_for_clustering/*.txt')
X = None
for file in label_files:
    with open(file, 'r') as f:
        l = f.readline().strip().split(' ')
        w = float(l[3])
        h = float(l[4])
        
        if X is None:
            X = np.array([[w, h]])
        else:
            X = np.vstack((X, np.array([w, h])))

model = KMeans(9)
# X_13 = X*13
# X_26 = X*26
# X_52 = X*52
X_416 = X*416
X = [X_416]

for x in X:

    model = model.fit(x)
       
    with open('./data/anchor_box_dims/scale.txt', 'a') as f:
        for i in range(model.cluster_centers_.shape[0]):
            print(str(model.cluster_centers_[i]))
            # f.write('\n')
            f.write(str(model.cluster_centers_[i]))
            f.write('\n')
              
        # f.write('\n')
        # f.write('\n')
        

