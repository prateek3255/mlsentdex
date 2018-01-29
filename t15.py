import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

# plt1=[2,6]
# plt2=[7,8]

# euclidean_distance=sqrt((plt1[0]-plt2[0])**2 + (plt1[1]-plt2[1])**2)

dataset={'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features=[5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],s=100,color=i)

# [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
#
# plt.scatter(new_features[0],new_features[1])
# plt.show()
def k_nearest_neighbors(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('k should be greater than total voting groups')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes=[i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    votes_result=Counter(votes).most_common(1)[0][0]
    return votes_result

result=k_nearest_neighbors(dataset,new_features,3)
print (result)
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]

plt.scatter(new_features[0],new_features[1],color=result)
plt.show()

