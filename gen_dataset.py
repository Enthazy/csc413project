import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import imageio
import numpy as np
filepath = "./data_for_model/velocities/classical/"
onlyfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]
print(len(onlyfiles))

def read_random_data_batch(filepath, batch_size):
    global onlyfiles
    TOTAL_NUMBER_OF_MIDI_FILES = len(onlyfiles)
    songs = []
    idxs = np.random.choice(TOTAL_NUMBER_OF_MIDI_FILES, batch_size)
    for idx in idxs:
        song = np.load(filepath+onlyfiles[idx])
        while song.shape[0] < 88:
            song = np.load(filepath+onlyfiles[np.random.randint(0, len(onlyfiles))])
        song = np.array(song)/127.0
        if song.shape[0] > 88:
            start = np.random.randint(0, song.shape[0]-88)
            song = song[start:start+88, :]
        song = song.reshape(88, 88, 1)
        songs.append(song)
    return np.array(songs), np.zeros(batch_size)

def gen_train_datasets(genre,size=10):
    images, labels = read_random_data_batch(filepath, size)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2])
    for idx in range(size):
        path = os.path.join('gandata',genre,'train', 'sample-{:06d}.png'.format(idx))
        imageio.imwrite(path, images[idx])
        print('Saved {}'.format(path))

def gen_test_datasets(genre, size=5):
    images, labels = read_random_data_batch(filepath, size)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2])
    for idx in range(size):
        path = os.path.join('gandata',genre,'test', 'sample-{:06d}.png'.format(idx))
        imageio.imwrite(path, images[idx])
        print('Saved {}'.format(path))

def display_images(images):
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2])
    random_indices = np.random.choice(images.shape[0], 8)
    idx = 0
    plt.figure(figsize=(8, 6))
    for i in range(2):
        for j in range(4):
            plt.subplot(2, 4, 4*i+j+1)
            plt.imshow(images[random_indices[idx]])
            idx += 1
    plt.show()

def display_pair(original, generated):
    original = original.reshape(original.shape[0], original.shape[1], original.shape[2])
    generated = generated.reshape(generated.shape[0], generated.shape[1], generated.shape[2])
    random_indices = np.random.choice(original.shape[0], 6)
    idx = 0
    plt.figure(figsize=(8, 6))
    for i in range(3):
        for j in range(2):
            plt.subplot(3, 4, 4 * i + 2 * j + 1)
            plt.imshow(original[random_indices[idx]])
            plt.subplot(3, 4, 4 * i + 2 * j + 2)
            plt.imshow(generated[random_indices[idx]])
            idx += 1
    plt.show()



# songs, labels = read_random_data_batch(filepath, 64)
# print(songs.shape)
# display_pair(songs, songs)

if __name__ == '__main__':
    gen_train_datasets("classical", 10)
    gen_test_datasets('classical',5)