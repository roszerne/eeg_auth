import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import json

n = 109 # Number of categories (users)
T = 160  # Length of one window (in samples)
delta = 4  # Sliding step
Delta = 8 # Sampling step
eta = 20 # Number of overlapped segments for one sample
C = 32 # Number of channels
input_shape = (eta, T, C) # input shape of one sample

samples_per_file_train = 1000
samples_per_file_valid = 170
list_IDs_train = np.arange(n * samples_per_file_train) # Indices of all samples
list_IDs_valid = np.arange(n * samples_per_file_valid) # Indices of all samples
id_to_sample_index_valid = [] # Based on id of sample, determine the original file
id_to_sample_index_train = [] # Based on id of sample, determine the original file

for file_index in range(n):
    id_to_sample_index_train.extend([file_index] * samples_per_file_train)
    id_to_sample_index_valid.extend([file_index] * samples_per_file_valid)

def format_filename(n):
    n += 1
    formatted_number = f"{n:03d}"  
    return f"S{formatted_number}R01_fp16.npy"

def get_file_index(id_to_sample_index, line_number):
    return id_to_sample_index[line_number]

def plot_training_history(history):

    # Retrieve a list of accuracy and loss metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(acc)
    print(val_acc)
    print(loss)
    print(val_loss)

     # Wyciągnij historię metryk
    history_dict = history.history

    # Zapisz historię do pliku JSON
    with open('history.json', 'w') as f:
        json.dump(history_dict, f)

    epochs = range(1, len(acc) + 1)
    
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, id_to_sample_index, samples_per_file, is_valid, batch_size=64, dim=(eta, T, C),
                 n_classes=n, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.id_to_sample_index = id_to_sample_index
        self.samples_per_file = samples_per_file
        self.is_valid = is_valid
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.list_IDs
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            file_num = get_file_index(self.id_to_sample_index, ID)
            file_name = format_filename(file_num)
            #array = np.load('shuffled/data_3/train/' + file_name, mmap_mode='r')
            array = np.load('data_32/' + file_name, mmap_mode='r')
            # print("SHAPE: ")
            # print(array.shape)
            # print("SHAPE: ")
            if self.is_valid:
                selected_array = array[samples_per_file_train + (ID % self.samples_per_file), :, :]
            else:
                selected_array = array[ID % self.samples_per_file, :, :]

            X[i,]= selected_array[np.newaxis, :, :]     
            # Store class
            y[i] = file_num

        return X, y#keras.utils.to_categorical(y, num_classes=self.n_classes)
    
training_generator = DataGenerator(list_IDs = list_IDs_train, id_to_sample_index = id_to_sample_index_train, samples_per_file = samples_per_file_train, is_valid = False)
validation_generator = DataGenerator(list_IDs = list_IDs_valid, id_to_sample_index = id_to_sample_index_valid, samples_per_file = samples_per_file_valid, is_valid = True)

model = Sequential()

# Convolutional layers
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', strides=(1, 1)))

# Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))  # Dropout layer with dropout rate of 0.5
model.add(Dense(n, activation='softmax'))

model.summary()

optimizer = RMSprop(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history =  model.fit_generator(generator = training_generator, validation_data = validation_generator, epochs = 10)
print("History: ")
print(history)

model.save('my_model.h5')
model.save_weights('my_model_weights.h5')
model.save('my_model.keras')

history_dict = history.history

# Saving data to a text file
with open('training_history.txt', 'w') as file:
    for key, values in history_dict.items():
        file.write(f"{key}: {values}\n")

loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# Display the model's structure to identify the layers to remove
model.summary()

# Remove the last layers - in this example, we remove the last 3 layers
# Build a new model from the beginning up to the layer before the ones to be removed
new_model = Model(inputs=model.input, outputs=model.layers[-4].output)

# Display the new model's structure to ensure the layers have been removed
new_model.summary()

# Save the modified model
new_model.save('fingerprinting.keras')