from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_fer2013_from_folders(train_dir="data/train", test_dir="data/test", target_size=(48, 48), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, test_gen