import os
from image_encoding import encoding_procedure
import numpy as np

path = os.path.join('images/')
images = os.listdir(path)
test_images = images[:1000]
train_images = images[1000:]
test_v = 'data_files/test_vector.npy'
train_v = 'data_files/train_vector.npy'

# Извлечение признаков изображения с помощью VGG16 готовой модели до
# последнего слоя не включая слой с softmax активатора
train_image_vector, test_image_vector = encoding_procedure(test_v,
                                                           train_v,
                                                           path,
                                                           test_images,
                                                           train_images)


