import logging
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import numpy as np


logger = logging.getLogger(__name__)


def train_model(data, units=100, **params):

    # TODO: this loads everything on memory
    x_seqs = np.array(
        [to_categorical(x, num_classes=data.vocab_size) for x in data.X]
    )
    y_seqs = np.array(
        [to_categorical(y, num_classes=data.vocab_size) for y in data.Y]
    )

    model = Sequential()
    model.add(
        keras.layers.GRU(units, input_shape=(data.seq_length, data.vocab_size))
    )
    model.add(Dense(data.vocab_size, activation='softmax'))
    logger.debug(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_seqs, y_seqs, verbose=2, **params)
    return model


def encode_input(txt, char_map):
    encoded = [char_map[c] for c in txt]
    encoded = to_categorical(encoded, num_classes=len(char_map.keys()))
    return encoded.reshape(1, encoded.shape[0], encoded.shape[1])


def generate_sequence(model, data, seed_txt, total):
    in_text = seed_txt

    for _ in range(total):
        
        # Encode input properly
        encoded = [data.char_to_id[c] for c in in_text]
        encoded = to_categorical(encoded, num_classes=data.vocab_size)
        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])

        # Restrict input to correct size
        encoded = encoded[:, -data.seq_length:, :]
        
        # Get output randomly according to softmax outputs
        outputs = model.predict(encoded, verbose=0)
        chosen = np.random.choice(range(data.vocab_size), p=outputs.ravel())
        new_char = data.id_to_char[chosen]
        
        in_text = in_text + new_char
        
    return in_text
