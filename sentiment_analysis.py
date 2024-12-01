import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

dataset = tf.keras.utils.get_file(origin="https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz", fname="stack_overflow_16k", extract=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.dirname(dataset)
train_dir = os.path.join(dataset_dir, 'train')
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory('train', batch_size=batch_size, validation_split=0.1, subset='training', seed=seed)
raw_val_ds = tf.keras.utils.text_dataset_from_directory('train', batch_size=batch_size, validation_split=0.1, subset='validation', seed=seed)
raw_test_ds = tf.keras.utils.text_dataset_from_directory('test', batch_size=batch_size)

max_features = 15000
sequence_length = 400
vectorize_layer = layers.TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
    layers.Embedding(max_features, 40),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)])

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

epochs = 75
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()