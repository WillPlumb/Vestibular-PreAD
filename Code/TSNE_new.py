import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import seaborn as sns
import pandas

features =[[' End Error', ' Average Z Acceleration', 'Average X Gyro rate of change per 0.1 seconds'],
           [' End Error', 'Age', 'Occupation'],
           [' End Error', ' Largest Magnitude Z Gyro Value']]

trial = [5,'7a','8a']#[1,2,3,4,5,6,'7a','8a',9]
for k in trial:
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    df =pandas.read_csv(path)
    df = df[df[' Trial']==str(k)]
    geneTarget = df['Gene Type']
    genearray = geneTarget.values
    colour=[]
    for g in genearray:
        if g==0:
            colour.append("g")
        else:
            colour.append("r")
    position = trial.index(k)
    df = df[features[position]]
    array = df.values
    #tsne = manifold.TSNE(n_components=3,random_state=1).fit_transform(array)
    tsne = manifold.TSNE(n_components=2,random_state=1).fit_transform(array)
    fig = plt.figure()
    title = 'Trial ' + str(k)
    fig.suptitle(title)
    #ax = fig.gca(projection='3d')
    #ax.scatter3D(tsne[:,0],tsne[:,1],tsne[:,2],c=colour)
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    #ax.set_zticklabels([])
    plt.scatter(tsne[:,0],tsne[:,1],c=colour)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    
