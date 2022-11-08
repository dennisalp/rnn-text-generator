import os
from pdb import set_trace as st
from utils import read_data, train_mod, gen_txt, load_model


class Bot():

    def __init__(self, mod, dat):

        self.seq_length = 10
        self.batch_size = 64
        self.buffer_size = 10000
        self.embedding_dim = 8
        self.rnn_units = 512
        self.epochs = 10

        self.dataset, self.ids2chars, self.chars2ids = read_data(dat, self.batch_size, self.seq_length, self.buffer_size)
        self.model = load_model(mod)

    def gen_txt(self, init_str, len_out_str, temperature=1.0):
        return gen_txt(self.model, init_str, len_out_str, self.ids2chars, self.chars2ids, temperature)


def main():
    bot = Bot()
    print(bot.gen_txt('hey whats up?', 42))


if __name__ == '__main__':
    main()
