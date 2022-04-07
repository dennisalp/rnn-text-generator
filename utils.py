import tensorflow as tf

from model import OneStepModel, CustomTrainingModel


def read_data(input_path, batch_size, seq_length, buffer_size):
    
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    text = open(input_path, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    chars2ids = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    ids2chars = tf.keras.layers.StringLookup(vocabulary=chars2ids.get_vocabulary(), invert=True, mask_token=None)

    all_ids = chars2ids(tf.strings.unicode_split(text, 'UTF-8'))
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
    

    history = model.fit(dataset, epochs=epochs)
    
    return model


def gen_txt(model, init_str, len_out_str, ids2chars, chars2ids, temperature=1.0):

    model = OneStepModel(model, ids2chars, chars2ids, temperature=temperature)
    
    states = None
    next_char = tf.constant([init_str])
    result = [next_char]

    for n in range(len_out_str):
        next_char, states = model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    print(result[0].numpy().decode('utf-8'))
