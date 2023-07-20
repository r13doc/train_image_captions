import random
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from gen_sbow import data, dataset, dataset_2, inverse_vocab, vocabula_size
import numpy as np

batch_size = 64
embedding_size = 1000
window_size = 1

epochs = 5
negative_samples = 4
valid_size = 16
valid_window = 250



vocabulary_size = vocabula_size

# K.clear_session()
#
input_1 = tf.keras.layers.Input(shape=())
input_2 = tf.keras.layers.Input(shape=(window_size*2,))

contex_embedding_layer = tf.keras.layers.Embedding(input_dim=vocabulary_size+1,
                                                   output_dim=embedding_size,
                                                   name='context_embedding')

target_embedding_layer = tf.keras.layers.Embedding(input_dim=vocabulary_size+1,
                                                   output_dim=embedding_size,
                                                   name='target_embedding')
contex_out = contex_embedding_layer(input_2)
target_out = target_embedding_layer(input_1)
# Берем среднее между контекстными словами, создаем размерность [None, embedding_size]
mean_contex_out = tf.keras.layers.Lambda(lambda x:
                                         tf.reduce_mean(x, axis=1))(contex_out)
# Расчитываем скалярное произведение векторов контекстного и целевого
out = tf.keras.layers.Dot(axes=-1)([mean_contex_out, target_out])

cbow_model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=out,
                                   name='cbow_model')
cbow_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   optimizer='adam',
                   metrics=['accuracy'])


class ValidCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, voc_size):
        self.model = model
        self.voc_size = voc_size

    def valid_check(self):
        # используем веса контекстных данных
        embedding_weights = self.model.get_layer('context_embedding').get_weights()[0]
        normalized_embeddings = embedding_weights / np.sqrt(embedding_weights ** 2, axis=1, keepdims=True)
        # Создаем тестовый набор слов из общего количества
        valid_ids = np.random.choice(self.voc_size, 16)
        # Получаем веса по набору данным
        valid_embeddings = normalized_embeddings[valid_ids, :]
        # Находим схожие веса между валидацией и контекстом V x d (d x D) => V x D
        top_k = 5
        similarity = np.dot(valid_embeddings, normalized_embeddings.T)
        # Инвертируем матрицу в отрицательную
        # Пропускаем первый номер
        simil_top_k = np.argsort(-similarity, axis=1)[:, 1: top_k+1]
        return simil_top_k


valid_callback = ValidCallback(cbow_model, vocabula_size)


# class ValidationCallback(tf.keras.callbacks.Callback):
#
#     def __init__(self, valid_term_ids, model_with_embeddings, tokenizer):
#         self.valid_term_ids = valid_term_ids
#         self.model_with_embeddings = model_with_embeddings
#         self.tokenizer = tokenizer
#
#         super().__init__()
#
#     def on_epoch_end(self, epoch, logs=None):
#         """ Validation logic """
#
#         # We will use context embeddings to get the most similar words
#         # Other strategies include: using target embeddings, mean embeddings after avaraging context/target
#         embedding_weights = self.model_with_embeddings.get_layer("context_embedding").get_weights()[0]
#         normalized_embeddings = embedding_weights / np.sqrt(np.sum(embedding_weights ** 2, axis=1, keepdims=True))
#
#         # Get the embeddings corresponding to valid_term_ids
#         valid_embeddings = normalized_embeddings[self.valid_term_ids, :]
#
#         # Compute the similarity between valid_term_ids and all the embeddings
#         # V x d (d x D) => V x D
#         top_k = 5  # Top k items will be displayed
#         similarity = np.dot(valid_embeddings, normalized_embeddings.T)
#
#         # Invert similarity matrix to negative
#         # Ignore the first one because that would be the same word as the probe word
#         similarity_top_k = np.argsort(-similarity, axis=1)[:, 1: top_k + 1]
#
#         # Print the output
#         for i, term_id in enumerate(valid_term_ids):
#             similar_word_str = ', '.join([self.tokenizer.index_word[j] for j in similarity_top_k[i, :] if j >= 1])
#             print(f"{self.tokenizer.index_word[term_id]}: {similar_word_str}")
#
#         print('\n')
#
#
# cbow_validation_callback = ValidationCallback(valid_term_ids, cbow_model, tokenizer)
#
# for ei in range(epochs):
#     print(f"Epoch: {ei+1}/{epochs} started")
#     news_cbow_gen = cbow_data_generator(news_sequences, window_size, batch_size, negative_samples)
#     cbow_model.fit(
#         news_cbow_gen,
#         epochs=1,
#         callbacks=cbow_validation_callback,
#
cbow_model.fit(dataset_2, epochs=5)
cbow_model.ge