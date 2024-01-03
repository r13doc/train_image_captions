import os
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# имена классов для тестирования
with open('class_names.txt', 'r') as clas:
    class_names = clas.readlines()
class_names = [i.replace('\n', '') for i in class_names]


def encoding_procedure(test_v, train_v, path_file, test_img, train_img):
    # Если данные обработаны мы просто загружаем их
    if os.path.exists(test_v) and os.path.exists(train_v):
        print('Подготовленные векторы модели загружены')
        train_vec = np.load(train_v)
        test_vec = np.load(test_v)
        return train_vec, test_vec

    else:
        # Обработка изображений
        print("Обработанные моделью векторы изображения не обнаружены")

        def prepare_image(img):
            files = path_file + img
            read = tf.io.read_file(files)
            im = tf.io.decode_jpeg(read, channels=3)
            im = tf.image.resize(im, [224, 224])
            im = tf.cast(im, tf.float32)
            image_pre = preprocess_input(im)
            return image_pre

        print("Загрузка модели!!")
        model = VGG16(weights='imagenet',
                      classifier_activation=None)

        # Сохранение обработанных изображений в файл numpy
        def vectorized(data, file_numpy):
            for ind, i in enumerate(data):
                prepare = prepare_image(i)
                im = tf.reshape(prepare, shape=(1, 224, 224, 3))
                pred = model.predict(im, verbose=0)
                if ind == 0:
                    test_train = np.array(pred)
                else:
                    test_train = np.append(test_train, pred, axis=0)
            np.save(file_numpy, test_train)

        print("Тестовая выборка")
        vectorized(test_img, test_v)
        print("Тренировочная выборка")
        vectorized(train_img, train_v)

        test_vec = np.load(test_v)
        train_vec = np.load(train_v)
        return train_vec, test_vec


# Тестирование данных
if __name__ == '__main__':
    from main import test_images, path, test_v

    test_vectors = np.load(test_v)

    for _ in range(15):
        number_image = np.random.choice(len(test_images), 1)
        pred_vector = test_vectors[number_image]
        file = test_images[number_image[0]]
        image = plt.imread(path + file)
        softmax = tf.nn.softmax(pred_vector)
        prev, prev2 = np.argmax(softmax), np.max(softmax)
        print(prev, prev2, class_names[prev])
        plt.imshow(image)
        plt.show()
        plt.pause(2.8)


