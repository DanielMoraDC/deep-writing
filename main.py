from input_data import get_sequences 
from model import train_model, generate_sequence
import logging

logging.basicConfig(
    level=logging.DEBUG
)


if __name__ == '__main__':

    data = get_sequences(length=10, subset=50000)
    model = train_model(data, units=80, epochs=20)

    seed_txt = 'The horse '.lower()
    total = 50
    generated = generate_sequence(model, data, seed_txt, total)

    print(seed_txt)
    print(generated)
