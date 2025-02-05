import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# Constants for the dataset and training
DATASET_DIR = "./Market-1501-v15.09.15/"
IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 20

# 1. Dataset Preparation
def load_dataset(dataset_dir):
    train_dir = os.path.join(dataset_dir, "bounding_box_train")
    test_dir = os.path.join(dataset_dir, "bounding_box_test")

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       horizontal_flip=True,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    return train_generator, test_generator

# 2. Model Creation
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(751, activation='softmax')  # 751 classes in Market-1501
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Training the Model
def train_cnn_model(model, train_generator, test_generator):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=test_generator.samples // BATCH_SIZE
    )
    return history

# 4. Evaluation and Metrics
def evaluate_model(model, test_generator):
    start_time = time.time()
    predictions = model.predict(test_generator)
    inference_time = time.time() - start_time

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    accuracy = accuracy_score(true_classes, predicted_classes)

    return accuracy, inference_time

# Main Function to Execute the Pipeline
def main():
    print("Loading dataset...")
    train_generator, test_generator = load_dataset(DATASET_DIR)

    print("Creating CNN model...")
    model = create_cnn_model()

    print("Training CNN model...")
    train_cnn_model(model, train_generator, test_generator)

    print("Evaluating CNN model...")
    accuracy, inference_time = evaluate_model(model, test_generator)

    print(f"Accuracy: {accuracy}")
    print(f"Inference Time: {inference_time} seconds")

    print("Saving the trained model...")
    model.save("person_reid_cnn_model.h5")

# Run the pipeline
if __name__ == "__main__":
    main()