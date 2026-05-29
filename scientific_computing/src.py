import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 10
IMG_SIZE = 32  

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_and_preprocess_data():

    # Train set = 50000; Test set = 10000
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalization
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def build_augmentation_pipeline():
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")
    return augmentation


def model(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_CIFAR10")
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(model, x_train, y_train, x_test, y_test, augmentation, epochs=EPOCHS, batch_size=BATCH_SIZE):
    x_train_aug = augmentation(x_train, training=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="best_cnn_model.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    return test_loss, test_acc


def save_training_plots(history, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy theo Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss theo Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] Đã lưu biểu đồ tại: {plot_path}")
    plt.show()


def predict_samples(model, x_test, y_test, n=10):
    y_pred = model.predict(x_test[:n], verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test[:n], axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(x_test[i])
        true_cls = CLASS_NAMES[y_true_labels[i]]
        pred_cls = CLASS_NAMES[y_pred_labels[i]]
        color = "green" if true_cls == pred_cls else "red"
        axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}", color=color, fontsize=9)
        axes[i].axis("off")
    plt.suptitle("Kết quả dự đoán (xanh = đúng, đỏ = sai)", fontsize=12)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Kiểm tra GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] GPU available: {len(gpus) > 0} ({gpus})")

    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Augmentation
    augmentation = build_augmentation_pipeline()

    # Build model
    model = model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.summary()

    # Compile
    model = compile_model(model, learning_rate=LEARNING_RATE)

    # Train
    history = train_model(model, x_train, y_train, x_test, y_test, augmentation)

    # Evaluate
    evaluate_model(model, x_test, y_test)

    save_training_plots(history)
    predict_samples(model, x_test, y_test)