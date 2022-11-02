import tensorflow as tf


""" 
Note: For training you could use a keras.Sequential model here. To
generate text later you'll need to manage the RNN's internal
state. It's simpler to include the state input and output options
upfront, than it is to rearrange the model architecture later.
"""
class BaseModel(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.gr2 = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)

        if states is None:
            states = []
            states.append(self.gru.get_initial_state(x))
            states.append(self.gr2.get_initial_state(x))
        x, states[0] = self.gru(x, initial_state=states[0], training=training)
        x, states[1] = self.gr2(x, initial_state=states[1], training=training)

        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStepModel(tf.keras.Model):
    
    def __init__(self, model, ids2chars, chars2ids, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.ids2chars = ids2chars
        self.chars2ids = chars2ids

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.chars2ids(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(chars2ids.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.split(inputs, ' ')
        input_ids = self.chars2ids(input_chars).to_tensor()
        if type(states) is list: states = states.copy()
        
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
#         predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.ids2chars(predicted_ids)
        return predicted_chars, states


class CustomTrainingModel(BaseModel):

    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss}
