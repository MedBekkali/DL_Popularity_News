import tensorflow as tf
def build_mlp_classifier(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="x")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="p_viral")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp_classifier")

def build_mlp_regressor(input_dim: int, output_dim: int = 3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="x")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dim, activation="linear", name="y_hat")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp_regressor")