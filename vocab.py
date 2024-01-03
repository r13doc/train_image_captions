import numpy as np
import tensorflow as tf
import os
import json


cap_val = os.path.join('annotations', 'captions_val2014.json')
text_annot = 'data_files/text_annotations.npy'

with open(cap_val, 'r') as f:
    cap_annot = json.load(f)

auto = tf.data.AUTOTUNE
max_len_sentence = 12


def prepare_annotations():
    if os.path.exists(text_annot):
        print("Подгрузка данных из файла аннотаций text_annotations.npy в vocab")
        print("================================================================", '\n')
        r_data = np.load(text_annot)
        return r_data

    else:
        def data_prepare():
            files = []
            for b in cap_annot['annotations']:
                for i in cap_annot['images']:
                    if i['id'] == b['image_id']:
                        files.append([i['file_name'], b['caption']])
            file = np.array(files)
            with open(text_annot, 'wb') as annot:
                np.save(annot, file)

        print("Подготовка данных, аннотаций в vocab")
        print("Создание файла аннотаций в data_files/text_annotations.npy")
        print("===========================================================", '\n')
        data_prepare()
        r_data = np.load(text_annot)
        return r_data


# определен метод, и количество слов в предложении
vectorize_layer = tf.keras.layers.TextVectorization(output_sequence_length=12)


# Подготовка данных для создания словаря(vocabulary)
def vocab(data):
    raw_data = tf.data.Dataset.from_tensor_slices(data)

    def prepare(some_data):
        return some_data[1]

    new_data = (raw_data
                .map(prepare, num_parallel_calls=auto)
                .cache())

    return raw_data, new_data


all_data = prepare_annotations()

ten_d, prepare_d = vocab(all_data)
vectorize_layer.adapt(prepare_d)

numbers_to_words = vectorize_layer.get_vocabulary()
n_vocab = vectorize_layer.vocabulary_size()

with open("data_files/numbers_to_words_n_vocab.npy", "wb") as files:
    np.save(files, numbers_to_words)
    np.save(files, n_vocab)

if __name__ == "__main__":
    print("\n")
    print("Тестирование vocab.py")
    print(f"количество слов в словаре {n_vocab}")
    for i in prepare_d.take(10):
        print(i.numpy().decode())
        print(vectorize_layer.call(i).numpy())
