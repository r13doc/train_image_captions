import tensorflow as tf
from embedding import raw_data
import random
import numpy as np
AUTOTUNE = tf.data.AUTOTUNE


def data_tr():
    def prep(x):
        pre = tf.strings.lower(x[1])
        split = tf.strings.split(pre)
        regex = tf.strings.regex_replace(split, "[^\w\s]", "")
        back = tf.strings.reduce_join(regex, axis=-1, separator=" ")
        return back

    data = (raw_data.take(1000)
            .map(prep, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(10))
    return data


data = data_tr()
vectorized_layer = tf.keras.layers.TextVectorization(output_sequence_length=12)
vectorized_layer.adapt(data)

inverse_vocab = vectorized_layer.get_vocabulary()
vocabula_size = vectorized_layer.vocabulary_size()

data_vectors = (data
                .map(vectorized_layer, num_parallel_calls=AUTOTUNE)
                .batch(100)
                .cache()
                .prefetch(100)
                .unbatch())

sequences = list(data_vectors.as_numpy_iterator())
# 'a bicycle replica with a clock as the front wheel'>

# Генерирует правильные контекстные слова сцелевым словом и 4 неправильных примера контекстных слов с целевым словом
def cbow_grams(seq, vocab_size, window_size,
               negat_sampled, sampling_table=None):
    targets, contexts, labels = [], [], []
    for sequence in seq:
        for ind, wi in enumerate(sequence):
            if not wi or ind < window_size or ind +1 > len(sequence) - window_size:
                continue
            if sampling_table is not None:
                if sampling_table[wi] < random.random():
                    #print (f"sampling table {sampling_table[wi]}, < random.random {random.random()}")
                    continue
            windows_start = max(0, ind-window_size)
            windows_end = min(len(sequence), ind+window_size+1)
            context_words = [wj for ind_j, wj in enumerate(
                sequence[windows_start:windows_end]) if ind_j+windows_start != ind]
            target_word = wi
            # print(sequence)
            # print([inverse_vocab[i] for i in sequence])
            # print(f'индекс {ind}, слово { wi}')
            # print(f'начало окна индекс {windows_start}, конец окна индекс {windows_end}')
            # print(f'контекстные слова {context_words}')
            # print(f'целевое слово {target_word}')
            context_classes = tf.expand_dims(tf.constant(context_words, dtype="int64"), 0)
            #print(f'контекст_класс {context_classes}')
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_classes,
                num_true=window_size * 2,
                num_sampled=negat_sampled,
                unique=True,
                range_max=vocab_size+1,
                name="negative_sampling")
            # print(f'отрицательная выборка кандидатов {negative_sampling_candidates}')
            # Build context and label vectors (for one target word)
            negative_targets = negative_sampling_candidates.numpy().tolist()
            #print(negative_targets, wi)
            target = [target_word] + negative_targets
            label = [1] + [0] * 4 # negative_samples
            # print(target)
            # print(label)
            # Append each element from the training example to global lists.
            targets.extend(target)
            contexts.extend([context_words] * (4 + 1)) # 4 negative_samples
            labels.extend(label)
            # print(f'targets {targets}, contexts {contexts}, labels {labels}')

    target, context, label = np.array(targets), np.array(contexts), np.array(labels)
    #couples = list(zip(targets, contexts))
    return target, context, label


sampl_table = tf.keras.preprocessing.sequence.make_sampling_table(vocabula_size)


targets, contexts, labels = cbow_grams(seq=sequences,
                                       vocab_size=vocabula_size,
                                       window_size=1,
                                       negat_sampled=4,
                                       sampling_table=None)


dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset_2 = dataset.batch(12, drop_remainder=True).prefetch(AUTOTUNE)
data = dataset.batch(12).cache().prefetch(AUTOTUNE)


