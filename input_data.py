from pathlib import Path
import urllib.request
import os
import logging
import collections
from collections import namedtuple
import numpy as np


PATH = os.path.join(str(Path.home()), '.book.txt')
URL = 'https://www.gutenberg.org/files/1399/1399-0.txt'

logger = logging.getLogger(__name__)


Data = collections.namedtuple(
    'Data',
    [
        'char_to_id', 'id_to_char', 'seq_length', 'vocab_size',
        'total', 'X', 'Y'
    ]
)


def is_downloaded():
    return os.path.isfile(PATH)


def download_file():
    urllib.request.urlretrieve(URL, PATH)


def get_sequences(length=10, skip=1055, subset=None):
    """
    Returns the set of sequences of the given length in the selected
    book.
    """

    if not is_downloaded():
        logger.info('Data not found. Downloading book...')
        download_file()
    
    data = open(PATH, 'r').read()
    # Let's erase the header. Default is for Anna Karenina
    data = data[skip:].lower()

    if subset is not None:
        data = data[:subset]

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    logger.info('Found %d characters' % data_size)
    logger.info('Data has %d characters' % vocab_size)

    char_to_ix = {ch: i for i,ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i,ch in enumerate(sorted(chars))}
    
    # Extract sequences of given length from data and encoded
    # them into the corresponding id of the chars
    sequences = list()
    for i in range(length, len(data)):
        seq = data[i-length:i+1]
        encoded = [char_to_ix[c] for c in seq]
        sequences.append(encoded)
    logger.info('Found %d sequences of length %d' % (len(sequences), length))

    sequences = np.asarray(sequences)
    return Data(
        X=sequences[:,:-1],
        Y=sequences[:,-1],
        char_to_id=char_to_ix,
        id_to_char=ix_to_char,
        vocab_size=vocab_size,
        seq_length=length,
        total=data_size
     )
