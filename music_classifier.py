from sklearn import neighbors,linear_model,svm,datasets
from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.colors import ListedColormap
import os
import sys
print(sys.path)





def func1():
    import glob
    import errno
    import librosa
    beats=[]
    tempos=[]
    types=[]

    #give path for the music folder use wav format
    path = r'.\*.wav'
    files = glob.glob(path)
    for name in files:

        try:
            with open(name) as f:
                y, sr = librosa.load(name)
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                print(name)
                print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
                tempos.append(tempo)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                beat_times=int(sum(beat_times)/len(beat_times))
                print(beat_times)
                beats.append(beat_times)
                types.append(0)
                
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise

    df2=[beats,tempos,types]
   
    return df2



def main():
    os.chdir("C:/Users/Shrawan/Desktop/music_classifier/fastwav")
    from btf import func123

    df1=func123()
    #print(df1)

    os.chdir("C:/Users/Shrawan/Desktop/music_classifier/slowwav")

    df2=func1()
    #print(df2)

    df1[0].extend(df2[0])
    df1[1].extend(df2[1])
    df1[2].extend(df2[2])

    #print(df1)


    x_beats=df1[0]
    x_tempo=df1[1]
    y=df1[2]



    x_train1=[x_beats,x_tempo]
    print(x_train1)
    print(type(x_train1))

    cvv=np.array(x_train1)
    
    dd=np.array(x_train1)
    dd=dd.transpose()
    x_train=dd.tolist()
    print(x_train)
    print(type(x_train))
    
    y_train=y

    clss=neighbors.KNeighborsClassifier(5)

    clss.fit(x_train,y_train)


    os.chdir("C:/Users/Shrawan/Desktop/music_classifier/testing")

    from testerbt import funct
 

    z=funct()
    #print(z)
    names=z.pop()
    hh=np.array(z)
    hh=hh.transpose()
    gg=hh.tolist()

    x_test=gg

    h = .02


    pred=clss.predict(x_test)
    #print(pred)
    #print(type(pred))
    pred=pred.tolist()
    #print(pred)
    #print(type(pred))

    print('#######################################################################')
    for i in range(len(pred)):
        if int(pred[i])==0:
            print(names[i],': Slow/Sad')
        elif int(pred[i])==1:
            print(names[i],':Fast/Party')

    #from sklearn.metrics import accuracy_score

    #print(accuracy_score(y_test,pred))




    '''X=cvv
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clss.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


    plt.show()'''
        

main()
                                                
