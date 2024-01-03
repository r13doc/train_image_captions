import tensorflow as tf
import numpy as np

from vocab import prepare_d, vectorize_layer
import random
import os

targ_cont_label = "data_files/targ_cont_label.npy"

# n_vocab = vectorize_layer.vocabulary_size()
# numbers_to_words = vectorize_layer.get_vocabulary()

with open("data_files/numbers_to_words_n_vocab.npy", 'rb') as f:
    _ = np.load(f)
    n_vocab = np.load(f)


# подготовка функции для c_bow алгоритма, размер окна слов указывать полностью
def cbow_gram(seq, voc_size, window_size=4,
              negative_samples=4):

    # seq = list(seq.as_numpy_iterator())[0].decode()
    seq = seq.numpy().decode()
    num = vectorize_layer.call(seq).numpy().tolist()

    all_target_words, all_context_words, labels = [], [], []

    window_start = int(window_size/2)
    window_end = int(len(num) - window_size/2)

    # целевые слова
    true_target_words = [i for ind, i in enumerate(num, start=1) if window_start < ind < window_end+1]

    # контекстные слова
    true_cont = []
    for ind_c, i_c in enumerate(num):
        if (window_start - 1) < ind_c < window_end:
            cont = num[ind_c - window_start:ind_c]
            cont.extend(num[ind_c+1:ind_c + window_start+1])
            true_cont.append(cont)
            # все контекстные слова
            all_context_words.extend([cont]*(negative_samples + 1))

    # отрицательные целевые слова
    negative_targets = []
    for contex_cl in true_cont:
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                        true_classes=tf.expand_dims(tf.constant(contex_cl, dtype="int64"), 0),
                        num_true=window_size * 1,
                        num_sampled=negative_samples,
                        unique=True,
                        range_max=voc_size,
                        name="negative_sampling")

        negative_targets.append(negative_sampling_candidates.numpy().tolist())
        # все метки отрицательные и положительные по порядку
        labels.extend([1] + [0] * negative_samples)

    # все целевые слова и отрицательные
    for t_t, n_t in zip(true_target_words, negative_targets):
        all_target_words.extend([t_t]+n_t)
    # определяем случайное число для всех данных
    seed = random.randint(0, 10e6)

    random.seed(seed)
    random.shuffle(all_target_words)

    random.seed(seed)
    random.shuffle(all_context_words)

    random.seed(seed)
    random.shuffle(labels)
    return all_target_words, all_context_words, labels


if os.path.exists(targ_cont_label):
    print("Подгрузка данных targ_cont_label.npy из sbow")
    print("=============================================", '\n')
    with open(targ_cont_label, 'rb') as f:
        target_w = np.load(f)
        context_w = np.load(f)
        lable_w = np.load(f)

else:
    # сохраняем результат в npy-файл
    def save_numpy():
        count = 0
        for ind, text in enumerate(prepare_d):
            count += 1
            if count % 20000 == 0:
                print(f"Подготовлены данные из {count} примеров, их всего {prepare_d.cardinality()}")
                #print(f"размер контекстных данных {count.size}")

            ta, co, la = cbow_gram(text, voc_size=n_vocab)
            if ind == 0:
                targ = np.array(ta, dtype='int32')
                contex = np.array(co, dtype='int32')
                lab = np.array(la, dtype='int32')
            else:
                targ = np.append(targ, ta)
                contex = np.append(contex, co, axis=0)
                lab = np.append(lab, la)

        with open(targ_cont_label, 'wb') as file:
            np.save(file, targ)
            np.save(file, contex)
            np.save(file, lab)

        return targ, contex, lab

    print("Подготовка данных targ_cont_label.npy из sbow")
    print("================================", '\n')
    target_w, context_w, lable_w = save_numpy()


if __name__ == '__main__':
    # проверка правильных целевых слов, контекста и меток,
    # созданных функцией cbow_gram
    for i in prepare_d.take(1):
        tar, cont, lab = cbow_gram(i, voc_size=n_vocab)
        print(i.numpy().decode())
        print(vectorize_layer.call(i).numpy().tolist())
        true_var = list(zip(tar, cont, lab))
        for t in true_var:
            if t[2] == 1:
                print(t)
