import os
from pdb import set_trace as st
from utils import read_data, train_mod, gen_txt


def main():
    
    input_path = os.path.join('dat', 'astrophysics.txt')
    pth = 'mod'
    init_str = 'what?'
    len_out_str = 2000
    seq_length = 100
    batch_size = 64
    buffer_size = 10000
    embedding_dim = 256
    rnn_units = 1024
    epochs = 30

    dataset, ids2chars, chars2ids = read_data(input_path, batch_size, seq_length, buffer_size)
    if os.path.isdir(pth):
        model = load_model(pth)
    else:
        model = train_mod(dataset, epochs, embedding_dim, rnn_units, ids2chars, chars2ids)
        model.save(pth)
    
    gen_txt(model, init_str, len_out_str, ids2chars, chars2ids, temperature=1.0)
    st()
    gen_txt(model, init_str, len_out_str, ids2chars, chars2ids, temperature=1.0)


if __name__ == '__main__':
    main()
