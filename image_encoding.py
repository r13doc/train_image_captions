import os
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# имена классов
with open('class_names.txt', 'r') as clas:
    class_names = clas.readlines()
class_names = [i.replace('\n', '') for i in class_names]

path = os.path.join('images/')
images = os.listdir(path)
test_images = images[:1000]
train_images = images[1000:]
test_v = 'test_vectorizer.npy'
train_v = 'train_vectorizer.npy'


if os.path.exists(test_v) and os.path.exists(train_v):
    print('Подготовленные векторы модели загружены')
    train_vector = np.load(train_v)
    test_vector = np.load(test_v)

else:
    # Обработка изображений
    def prepare_image(img):
        files = path + img
        read = tf.io.read_file(files)
        im = tf.io.decode_jpeg(read, channels=3)
        im = tf.image.resize(im, [224, 224])
        image_pre = preprocess_input(im)
        return image_pre

    print("Загрузка модели")
    model = VGG16(weights='imagenet',
                  classifier_activation=None)

    # Сохранение обработанных изображений в файл numpy
    def vectorized(data, check_npy):
        if os.path.exists(check_npy):
            if check_npy == train_v:
                print('Созданы тренировочные данные')
            else:
                pass
        else:
            for ind, i in enumerate(data):
                prepare = prepare_image(i)
                im = tf.reshape(prepare, shape=(1, 224, 224, 3))
                pred = model.predict(im, verbose=0)
                if ind == 0:
                    test_train = np.array(pred)
                else:
                    test_train = np.append(test_train, pred, axis=0)
            np.save(check_npy[:-4], test_train)


    vectorized(test_images, test_v)
    vectorized(train_images, train_v)

    train_vector = np.load(train_v)
    test_vector = np.load(test_v)


# Тестирование данных после модели
if __name__ == '__main__':
    count = 0

    for file, pred_vector in zip(test_images, test_vector):
        image = plt.imread(path + file)
        softmax = tf.nn.softmax(pred_vector)
        prev, prev2 = np.argmax(softmax), np.max(softmax)
        print(prev, prev2, class_names[prev])
        plt.imshow(image)
        plt.show()
        plt.pause(0.8)
        count += 1
        if count % 15 == 0:
            break
