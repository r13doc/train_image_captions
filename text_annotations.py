# Подготовка аннотаций относящихся к изображениям
import os
import pickle
import json
import re
from image_encoding import test_images, train_images

cap_val = os.path.join('annotations', 'captions_val2014.json')
annot_file = 'dict_words.pkl'

if os.path.exists(annot_file):
    # чтение сохраненных данных через pickle
    with open('dict_words.pkl', 'rb') as p:
        dictionary, reverse_dictionary, vocabulary_size = pickle.load(p)
        print(f'Длина vocabulary_size: {vocabulary_size}')

    with open('captions.pkl', 'rb') as p:
        captions = pickle.load(p)

else:
    def split(capt):
        return capt.split(' ')

    def prep(file):
        value = re.sub(r'[^\w\s]', '', file)
        value = value.replace('\n', '')
        return value.lower()

    def annotations(cap):
        images_caption = {}
        words = list()
        max_len = 0
        with open(cap, 'r') as f:
            cap_annot = json.load(f)
        for index in cap_annot['images']:
            images_caption[index['id']] = [(index['file_name'])]

        for index in cap_annot['annotations']:
            if index['image_id'] in images_caption:
                caption = prep(index['caption'])
                images_caption[index['image_id']].append(caption)
                # Определяем максимальное число слов в предложении
                if len(split(caption)) > max_len:
                    max_len = len(split(caption))
                # Обновляем список уникальных слов
                words = list(set(words).union(set(split(caption))))

        return images_caption, words, max_len

    # Подготовка подписей
    def repl_images(capt_im):
        new = {}
        for i in capt_im.values():
            new[i[0]] = i[1:]
        return new

    # подписи, все слова в подписях,
    im_captions, unique_words, orig_max_len = annotations(cap_val)
    captions = repl_images(im_captions)


    def dict_word(words):
        diction = {'НАЧ': 0, 'КОН': 1}
        for tg in words:
            diction[tg] = len(diction)
        # Create the reverse dictionary
        rev_dictionary = dict([(v, k) for k, v in diction.items()])
        voc_size = len(diction)
        return diction, rev_dictionary, voc_size


    dictionary, reverse_dictionary, vocabulary_size = dict_word(unique_words)
    with open('dict_words.pkl', 'wb') as p:
        pickle.dump((dict_word(unique_words)), p)
    with open('captions.pkl', 'wb') as p:
        pickle.dump(captions, p)

# Проверка аннотаций
if __name__ == '__main__':
    count = 0
    for r, d in zip(reverse_dictionary.items(), dictionary.items()):
        count += 1
        print(r, ' reverse_dictionary', '\n', d, 'dictionary')
        if count % 10 == 0:
            break
    print(f'Длина vocabulary size {vocabulary_size}')
