from keras.utils import img_to_array, load_img
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
import csv
import os
import numpy as np

def sigmoider(num):
    if num > 0:
        return 0
    else:
        return 1

train_data_folder = ["photos/training/side/photos", "photos/training/heads/photos", "photos/training/tails/photos"]
train_csv = ["photos/training/side/data.csv", "photos/training/heads/data.csv", "photos/training/tails/data.csv"]

training_x = []
training_y = []
train_ht = []
train_pic = []

RESOLUTION = 320
x_cord = []
y_cord = []
ht = []

for a in range(3):
    with open(train_csv[a], encoding='utf-8-sig') as file:
        for row in csv.reader(file, delimiter=';'):
            x_cord.append(row[0])
            y_cord.append(row[1])
            ht.append(row[2])

        for b, photo_files in enumerate(sorted(os.listdir(train_data_folder[a]))):
            arr = img_to_array(load_img(os.path.join(train_data_folder[a], str(photo_files)),
                                         target_size=(RESOLUTION, RESOLUTION)))
            train_pic.append(arr / 255)   

training_x = np.array(x_cord).astype(float)
training_y = np.array(y_cord).astype(float)
train_ht = np.array(ht).astype(int)
train_pic = np.array(train_pic).astype(float)

valid_data_folder = ["photos/validation/side/photos", "photos/validation/heads/photos", "photos/validation/tails/photos"]
valid_csv = ["photos/validation/side/data.csv", "photos/validation/heads/data.csv", "photos/validation/tails/data.csv"]
valid_x = []
valid_y = []
valid_ht = []
valid_pic = []

x_cord = []
y_cord = []
ht = []
for a in range(3):
    with open(valid_csv[a], encoding='utf-8-sig') as file:
        for row in csv.reader(file, delimiter=';'):
            x_cord.append(row[0])
            y_cord.append(row[1])
            ht.append(row[2])

        for b, photo_files in enumerate(sorted(os.listdir(valid_data_folder[a]))):
            arr = img_to_array(load_img(os.path.join(valid_data_folder[a], str(photo_files)),
                                         target_size=(RESOLUTION, RESOLUTION)))
            valid_pic.append(arr / 255)

valid_x = np.array(x_cord).astype(float)
valid_y = np.array(y_cord).astype(float)
valid_ht = np.array(ht).astype(int)
valid_pic = np.array(valid_pic).astype(float)

test_data_folder = ["photos/testing/side/photos", "photos/testing/heads/photos", "photos/testing/tails/photos"]
test_csv = ["photos/testing/side/data.csv", "photos/testing/heads/data.csv", "photos/testing/tails/data.csv"]
testing_x = []
testing_y = []
testing_ht = []
testing_pic = []

x_cord.clear()
y_cord.clear()
ht.clear()
for a in range(3):
    with open(test_csv[a], encoding='utf-8-sig') as file:
        for row in csv.reader(file, delimiter=';'):
            x_cord.append(row[0])
            y_cord.append(row[1])
            ht.append(row[2])

        for b, photo_files in enumerate(sorted(os.listdir(test_data_folder[a]))):
            arr = img_to_array(load_img(os.path.join(test_data_folder[a], str(photo_files)),
                                         target_size=(RESOLUTION, RESOLUTION)))
            testing_pic.append(arr / 255) 

testing_x = np.array(x_cord).astype(float)
testing_y = np.array(y_cord).astype(float)
testing_ht = np.array(ht).astype(int)
testing_pic = np.array(testing_pic).astype(float)

in_pic = Input(shape=(RESOLUTION, RESOLUTION, 3), name="in_pic")

conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(in_pic)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu")(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)

x_output = Dense(1, activation='linear', name='x_output')(flat)
y_output = Dense(1, activation='linear', name='y_output')(flat)

output_bool = Dense(1, activation='sigmoid', name="output_bool")(flat)
cnn = Model(inputs=in_pic, outputs=[x_output, y_output, output_bool])

cnn.compile(optimizer=Adam(), loss=['mse', 'mse', 'binary_crossentropy'],
             metrics={'x_output': 'mae', 'y_output': 'mae', 'output_bool': 'accuracy'})
cnn.fit(train_pic, [training_x, training_y, train_ht], epochs=30, batch_size=32,
         validation_data=(valid_pic, [valid_x, valid_y, valid_ht]))
eval = cnn.evaluate(testing_pic, [testing_x, testing_y, testing_ht])

print(f"Loss: {eval[0]}")
print(f"MAE for x: {eval[1]}")
print(f"MAE for y: {eval[2]}")
print(f"Accuracy: {eval[3]}")

pred = cnn.predict(testing_pic)
x_pred, y_pred, ht_pred = pred

for i in range(len(testing_pic)):
    print("Prediction",i+1,":")
    print("Predicted_x :",x_pred[i][0],"True_x :",testing_x[i])
    print("Predicted_y :",y_pred[i][0],"True_y :",testing_y[i])
    print("Predicted_orientation :",sigmoider(ht_pred[i]),"True_orientation :",testing_ht[i],"\n")