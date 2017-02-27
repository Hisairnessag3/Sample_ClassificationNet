from keras.models import Sequential
from keras.layers import *
import numpy
import tensorflow



seed = 6
numpy.random.seed(seed)

#Load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter= ",")
#Split input into X and Y
X = dataset[:,0:8]
Y = dataset[:,8]
#create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(5, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

model.add(Dropout(0.15, input_shape=(50,)))

#compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fit Model
model.fit(X, Y, nb_epoch=150, batch_size=10)
#evaluate
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
