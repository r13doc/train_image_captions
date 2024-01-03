import os
import vocab
import sbow
import embedding
from image_encoding import encoding_procedure

path = os.path.join('images/')
images = os.listdir(path)

# папка с изображениями coco_val_2014
test_images = images[:1000]
train_images = images[1000:]

# хранение данных в формате npy
test_v = 'data_files/test_vector.npy'
train_v = 'data_files/train_vector.npy'

os.makedirs("data_files", exist_ok=True)

# Извлечение признаков изображения с помощью VGG16 готовой модели до
# последнего слоя не включая слой с softmax активатора
print("Извлечение признаков изображения из данных в image_encoding")
print("================================================================", '\n')

train_image_vector, test_image_vector = encoding_procedure(test_v,
                                                           train_v,
                                                           path,
                                                           test_images,
                                                           train_images)
