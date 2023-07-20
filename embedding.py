import numpy as np
import re
import tensorflow as tf
import os
import json

from image_encoding import test_images, train_images
cap_val = os.path.join('annotations', 'captions_val2014.json')
with open(cap_val, 'r') as f:
    cap_annot = json.load(f)

auto = tf.data.AUTOTUNE
max_len_sentence = 12

# files = []
# f = np.array([])
# count = 0
# for i in cap_anot['images']:
#     count += 1
#     if count % 1000 == 0:
#         print(count)
#     for ind, b in enumerate(cap_anot['annotations']):
#         if i['id'] == b['image_id']:
#             #temp = np.append(temp, [b['caption']], axis=0)
#             files.append([i['file_name'], b['caption']])
        #if ind == 202653:
        #    files.append([i['file_name'], temp])
        #    temp = np.array([])

if os.path.exists('data_all.npy'):
    raw_data = np.load('data_all.npy')

else:

    def data_all():
        files = []
        for b in cap_annot['annotations']:
            for ind, i in enumerate(cap_annot['images']):
                if i['id'] == b['image_id']:
                    files.append([i['file_name'], b['caption']])
        file = np.array(files)
        return file

    raw_data = data_all()

    with open('data_all.npy', 'wb') as f:
        np.save(f, raw_data)

# Данные имя файла - аннотации к нему
raw_data = tf.data.Dataset.from_tensor_slices(raw_data)


def prepare_data():
    def trans(y, x):
        #prep = tf.strings.lower(x)
        #split = tf.strings.split(prep)
        #regex = tf.strings.regex_replace(split, "[^\w\s]", "")
        #x = tf.strings.reduce_join(regex, axis=-1, separator=" ")
        return x

    prep_data = (raw_data
                 .map(trans, num_parallel_calls=auto)
                 .cache()
                 .shuffle(30)
                 .prefetch(40))
    return prep_data


# def vocab(func):
#     def prep(y, x):
#         x = tf.strings.reduce_join(x, axis=-1, separator=" ")
#         return x
#
#     prep_voc = list(func
#                     .map(prep, num_parallel_calls=auto)
#                     .as_numpy_iterator())
#     return prep_voc
#
#
# orig_data = prepare_data()
# # vocab_prep = vocab(orig_data)
# text_vector = tf.keras.layers.TextVectorization()
# text_vector.adapt(vocab_prep)
# vocabulary_size = text_vector.vocabulary_size()


# def prepare_vocab():
#     words = []
#     def text(x):
#         return x[1]
#     prep_vocab = raw_data.map(text)
#     prep_vocab = list(prep_vocab.as_numpy_iterator())
#     #vocab = [i.decode('utf-8') for i in prep_vocab]
#     #split = [i.split()[:12] for i in vocab]
#     regex = []
#
#     # v = [[re.sub(r'[^\w\s]', '', i) for i in b] for b in vocab_list]
#     # v = [[re.sub(r'[^\w\s]', '', i.lower()) for i in b] for b in vocab_list]
#
#     #lower = tf.strings.lower(prep_vocab)
#     #splits = tf.strings.join(lower)
#     #spl = tf.strings.split(lower).numpy()
#     #regex = tf.strings.regex_replace(lower, '[\d|\s|"]', "")
#     #vocab = [i.decode('utf-8') for i in spl]
#     # for i in vocab:
#     #     if i in words:
#     #         continue
#     #     else:
#     #         words.append(i)
#     #vocab = [re.sub(r"[^\w\s\d]", '', i) for i in vocab]
#     return prep_vocab

