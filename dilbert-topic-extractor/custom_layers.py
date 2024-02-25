from tensorflow import shape, float32
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Metric
from tensorflow.keras.backend import cast, sum as K_sum, epsilon, equal, mean, batch_set_value
from tensorflow.math import argmax
from numpy import power, zeros, newaxis, sin, cos, arange, float32 as npFloat32

# Classi per gli strati Transformers personalizzati
class TransformerEncoderLayer(Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.3, regularizer=None, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.rate = rate
        self.regularizer = regularizer

        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)
        self.ffn = Sequential([
            Dense(self.dff, activation='relu', kernel_regularizer=regularizer),
            Dense(self.d_model, kernel_regularizer=regularizer)
        ])

        # Normalizzazione dello strato
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # Dropout per prevenire l'overfitting
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super(TransformerEncoderLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dff': self.dff,
            'rate': self.rate,
            'regularizer': self.regularizer
        })
        return config

    def call(self, inputs, training=False):
        # Attivazione del meccanismo di auto-attenzione
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Passaggio attraverso la rete feedforward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class PositionalEncoding(Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(self.position, self.d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / power(10000, (2 * (i // 2)) / npFloat32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            arange(position)[:, newaxis],
            arange(d_model)[newaxis, :],
            d_model
        )

        # Applicazione della funzione sinusoidale ai valori pari e della funzione coseno ai valori dispari
        angle_rads[:, 0::2] = sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[newaxis, ...]

        return cast(pos_encoding, dtype=float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :shape(inputs)[1], :]

class F1Score(Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes  # Numero di classi per la metrica F1

        # Inizializzazione delle variabili per tenere traccia dei veri positivi, falsi positivi e falsi negativi per ogni classe
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def get_config(self):
        config = super(F1Score, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = cast(argmax(y_pred, axis=-1), 'float32')
        y_true = cast(argmax(y_true, axis=-1), 'float32')

        def calculate_tp_fp_fn(y_true, y_pred):
            tp = zeros(self.num_classes, dtype='float32')
            fp = zeros(self.num_classes, dtype='float32')
            fn = zeros(self.num_classes, dtype='float32')

            for i in range(self.num_classes):
                y_true_class = equal(y_true, i)
                y_pred_class = equal(y_pred, i)

                tp = tp + cast(K_sum(cast(y_true_class & y_pred_class, 'float32')), 'float32')
                fp = fp + cast(K_sum(cast(y_pred_class & ~y_true_class, 'float32')), 'float32')
                fn = fn + cast(K_sum(cast(y_true_class & ~y_pred_class, 'float32')), 'float32')

            return tp, fp, fn

        tp, fp, fn = calculate_tp_fp_fn(y_true, y_pred)
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Calcola precisione e richiamo per ogni classe, quindi calcola la F1-Score
        precision = self.true_positives / (self.true_positives + self.false_positives + epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + epsilon())
        return mean(f1)  # Ritorna la media della F1-Score su tutte le classi

    def reset_state(self):
        # Reset delle variabili di stato alla fine di ogni epoca
        batch_set_value([(v, zeros(self.num_classes)) for v in self.variables])