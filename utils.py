import tensorflow as tf

from model import OneStepModel, CustomTrainingModel


chars2words = ['\n', '!', ',', '.', '?']
rm_chars = ['#', '$', '%', '&', '+', '/', ':', ';', '=', '>', '[', '`', 'é', '’', '“', '”', '(', ')']


def letters2words(text):    
    for char in chars2words:
        text = text.replace(char, ' {0:s} '.format(char))
    for char in rm_chars:
        text = text.replace(char, '')

    text = text.replace('  ', ' ')
    return text


def inverse_fmt(text):
    for char in chars2words:
        text = text.replace(' {0:s}'.format(char), char)
    text = text.replace('\n ', '\n')
    text = text.replace('  ', ' ')
    return text


def read_data(input_path, batch_size, seq_length, buffer_size):
    
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    text = open(input_path, 'rb').read().decode(encoding='utf-8').lower()
    text = letters2words(text).split(' ')
    vocab = sorted(set(text))

    chars2ids = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
    ids2chars = tf.keras.layers.StringLookup(vocabulary=chars2ids.get_vocabulary(), invert=True, mask_token=None)

#     all_ids = chars2ids(tf.strings.unicode_split(text, 'UTF-8'))
    all_ids = chars2ids(text)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    
    examples_per_epoch = len(text)//(seq_length+1)

    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)


    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, ids2chars, chars2ids


def train_mod(dataset, epochs, embedding_dim, rnn_units, ids2chars, chars2ids):
    model = CustomTrainingModel(
        vocab_size=len(chars2ids.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=3,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    history = model.fit(dataset, epochs=epochs, callbacks=[cb])
    
    return model


def gen_txt(model, init_str, len_out_str, ids2chars, chars2ids, temperature=1.0):

    model = OneStepModel(model, ids2chars, chars2ids, temperature=temperature)
    
    states = None
    if init_str[-1] != '\n': init_str += '\n'
    next_char = tf.constant([letters2words(init_str)])
    result = []

    for n in range(len_out_str):
        next_char, states = model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result, ' ')
    result = result[0].numpy().decode('utf-8')
    result = inverse_fmt(result)
    return result


def load_model(pth):
    old = tf.keras.models.load_model(pth)
    vs = old.layers[0].get_config()['input_dim']
    ed = old.layers[0].get_config()['output_dim']
    ru = old.layers[1].get_config()['units']
    
    new = CustomTrainingModel(
          vocab_size=vs,
          embedding_dim=ed,
          rnn_units=ru)
    
    new.build((None, vs))
    new.set_weights(old.get_weights())
    return new
