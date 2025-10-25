
import numpy as np
from activation import *
from layer import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits=datasets.load_digits()
images=digits.images
labels=digits.target

x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2)
x_train_reshaped=x_train.reshape(x_train.shape[0],-1)
x_test_reshaped=x_test.reshape(x_test.shape[0],-1)
## convert the test labels into arrays of probabilites
new_labels=[]
for label in y_train:
    probs=[0]*10
    probs[label]=1
    new_labels.append(probs)
y_train=np.array(new_labels)
y_train_reshaped=y_train.reshape(y_train.reshape[0],-1)
y_train=np.array(new_labels)
y_train_reshaped=y_train.reshape(y_train.reshape[0],-1)

def convert(classifier_predictions):
    predictions=[]
    for prediction in classifier_predictions:
        curr_pred=-1
        curr_prob=0
        for i,val in enumerate(prediction[0]):
            if val>curr_prob:
                curr_pred=i
                curr_prob=val
        prediction.append(curr_pred)



# def display_images(images,labels,title=None,predictions=None):


if __name__=="__main__":
    x_train_reshaped/=16
    x_test_reshaped/=16
    alpha=1e-2
    batch_size=32
    classifier=LayerList(Layer(64,28),Layer(28,10),activation=Softmax())
    classifier.fit(x_train_reshaped,y_train_reshaped,1000,alpha,batch_size,categorical_cross_entropy_loss)
    display_images(x_test,y_test,title="Numpy predictions",predictions=convert(classifier.predict(x_test_reshaped)))

