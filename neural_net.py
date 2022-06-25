import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import models, layers


train_file_path = "dataset/sign_mnist_train.csv"
test_file_path = "dataset/sign_mnist_test.csv"

train_data_df = pd.read_csv(train_file_path)
test_data_df = pd.read_csv(test_file_path)

# 0 indexed labels, i.e A = 0, B = 1 and so on...
train_labels = train_data_df["label"].to_numpy()
label_col = 'label'
train_data = train_data_df.loc[:,train_data_df.columns != label_col].to_numpy()
#print(train_labels.shape, train_data.shape)

test_labels = test_data_df["label"].to_numpy()
test_data = test_data_df.loc[:,test_data_df.columns != label_col].to_numpy()
#print(test_labels.shape, test_data.shape)


#to_print = train_data[14]
#to_print = to_print.reshape(28,28)
#plt.figure()
#plt.imshow(to_print)
#plt.grid(False)
#plt.show()

#print("Label value = ", train_labels[14])

'''
# This model is stored in hand_sign_model.h5, gives test accuracy of 70%

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(26, activation = 'softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()

model.fit(train_data, train_labels, epochs=30 ,batch_size=24)

predictions = model.predict(test_data)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test loss = ", test_loss, "Test_accuracy = ", test_acc*100, "%")
'''
new_train_data = train_data.reshape(-1,28,28)
new_test_data = test_data.reshape(-1,28,28)


# CNN model

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (28,28, 1)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation= 'sigmoid'))
model.add(layers.Dense(32, activation = 'sigmoid'))
model.add(layers.Dense(26, activation= 'softmax'))

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(new_train_data, train_labels, epochs=10, batch_size=24)










