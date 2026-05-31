import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def build_augmentation_pipeline():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")


def build_model(input_shape=(32, 32, 3), num_classes=10):

    inputs = keras.Input(shape=input_shape, name="input_image")

    data_augmentation = build_augmentation_pipeline()

    x = data_augmentation(inputs)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv1"
    )(x)

    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv2"
    )(x)

    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        name="conv3"
    )(x)

    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="fc1"
    )(x)

    x = layers.Dropout(0.5, name="dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output"
    )(x)

    return models.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_CIFAR10"
    )


def compile_model(model, learning_rate=LEARNING_RATE):

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
):

    callbacks = [

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ModelCheckpoint(
            filepath="best_cnn_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model(model, x_test, y_test):

    test_loss, test_acc = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")

    return test_loss, test_acc


def show_confusion_matrix(model, x_test, y_test):

    print("\n[INFO] Generating Confusion Matrix...")

    predictions = model.predict(
        x_test,
        verbose=0
    )

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_percent = (
        cm.astype("float")
        / cm.sum(axis=1)[:, np.newaxis]
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_percent,
        display_labels=CLASS_NAMES
    )

    disp.plot(
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
        ax=ax
    )

    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=150
    )

    print("[INFO] Saved: confusion_matrix.png")

    plt.show()


def save_training_plots(history):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    axes[0].plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )

    axes[0].plot(
        history.history["val_accuracy"],
        label="Val Accuracy"
    )

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history.history["loss"],
        label="Train Loss"
    )

    axes[1].plot(
        history.history["val_loss"],
        label="Val Loss"
    )

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        "training_history.png",
        dpi=150
    )

    plt.show()


def predict_samples(model, x_test, y_test, n=10):

    y_pred = model.predict(
        x_test[:n],
        verbose=0
    )

    y_pred_labels = np.argmax(
        y_pred,
        axis=1
    )

    y_true_labels = np.argmax(
        y_test[:n],
        axis=1
    )

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(15, 6)
    )

    axes = axes.flatten()

    for i in range(n):

        axes[i].imshow(x_test[i])

        true_cls = CLASS_NAMES[y_true_labels[i]]
        pred_cls = CLASS_NAMES[y_pred_labels[i]]

        color = (
            "green"
            if true_cls == pred_cls
            else "red"
        )

        axes[i].set_title(
            f"True: {true_cls}\nPred: {pred_cls}",
            color=color,
            fontsize=9
        )

        axes[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        "sample_predictions.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] GPU available: {len(gpus) > 0}")

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=NUM_CLASSES
    )

    model.summary()

    model = compile_model(
        model,
        learning_rate=LEARNING_RATE
    )

    history = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    evaluate_model(
        model,
        x_test,
        y_test
    )

    show_confusion_matrix(
        model,
        x_test,
        y_test
    )

    save_training_plots(history)

    predict_samples(
        model,
        x_test,
        y_test
    )