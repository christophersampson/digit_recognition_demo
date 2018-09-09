from os import listdir
from os.path import join
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

from keras.models import load_model
from confusion_matrix import create_confusion_matrix
from mpu.ml import one_hot2indices


def main():
    data_path = './data/mnist'

    training_path = join(data_path, 'training')
    test_path = join(data_path, 'testing')
    # training_path = './data/small_set'
    # test_path = './data/custom_set'
    # test_path = './data/mnist_png/testing'
    train(data_path=training_path)
    test(data_path=test_path, plot=True)


def pre_processing(image, im_size=(28, 28), invert=False):
    """

    Pre-processing function.
    Performs pre-processing operations an flattens image to a matrix with a single row.

    :param image: cv2 image
    :param im_size: image size tuple e.g. (28, 28)
    :return: x - Numpy array with single row

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, im_size)
    if invert:
        image = cv2.bitwise_not(image)
    _, image = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    image = np.array(image)
    return image, image.flatten()


def generate_matricies(categories_path, im_size=(28, 28), verbose=False):
    """
    Generates matrices from categorised image path.

    :param categories_path: path to folderised categories
    :param im_size: image size tuple e.g. (28, 28)
    :param verbose: verbosity
    :return: X design matrix, Y output matrix, category list

    """
    image_paths = []
    y_category = []
    X = []
    num_images = 0

    categories = listdir(categories_path)

    print('Loading images...')
    for category_val, category in enumerate(categories):

        category_images = listdir(join(categories_path, category))
        num_images += len(category_images)
        for image_name in category_images:
            image_path = join(categories_path, category, image_name)
            image_paths.append(image_path)

            image = cv2.imread(image_path)

            _, flattened_image = pre_processing(image, im_size)
            X.append(flattened_image)

            y_category.append(category_val)

    y_category = np.array(y_category)
    X = np.array(X)
    Y = to_categorical(y_category)

    if verbose:
        print(Y.shape)
        print(X.shape)

    return X, Y, categories


def train(data_path, hidden_layer_sizes=[784, 800, 10],
          im_size=(28, 28), dropout_fraction=0.2,
          random_seed=1, epochs=25, val_fraction=0.15,
          model_name='digit.keras'):
    """

  Trains feedforward neural network.
    Stores model.

    :param data_path: Path to training data.
    :param hidden_layer_sizes: List of hidden layer neuron sizes.
    :param im_size: Tuple of image size. Images are resized to this.
    :param dropout_fraction: Dropout fraction.
    :param random_seed: Random seed to use for validation split.
    :param epochs: Number of training iterations.
    :param val_fraction: Validation split fraction
    :param model_name: Saved model name. Can also be a path

    """

    # Get design and output matrices for training
    (X, Y, categories) = generate_matricies(data_path, im_size=im_size)

    # Explicitly split validation set.
    (train_X, validation_X, train_Y, validation_Y) = train_test_split(X, Y, test_size=val_fraction,
                                                                      random_state=random_seed)

    # Initialise model
    model = Sequential()

    # First layer
    model.add(Dense(hidden_layer_sizes[0], activation='relu', input_dim=train_X.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_fraction))

    # Deeper layers
    if len(hidden_layer_sizes) > 1:
        for layerNum in hidden_layer_sizes[1:]:
            model.add(Dense(layerNum, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_fraction))

    # Softmax layer
    model.add(Dense(train_Y.shape[1], activation='softmax'))

    model.compile(loss="binary_crossentropy", optimizer='sgd',
                  metrics=["accuracy"])

    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_acc',
                              min_delta=1e-3, patience=3,
                              verbose=1, mode='auto')
    callbacks_list = [early_stop]

    # Model training
    model.fit(train_X, train_Y, validation_data=(validation_X, validation_Y),
              epochs=epochs, batch_size=128, verbose=2,
              callbacks=callbacks_list)

    # Save the model
    print("Saving model..")
    model.save(model_name)
    print('Done.')


def test(data_path, model_name='digit.keras', plot=True):
    # Get design and output matrices for testing
    (test_X, test_Y, categories) = generate_matricies(data_path)

    model = load_model(model_name)

    # show the accuracy on the testing set
    print("Evaluating the testing set...")
    (loss, accuracy) = model.evaluate(test_X, test_Y,
                                      batch_size=128, verbose=2)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    if plot:
        pred_scores = model.predict(test_X,
                                    batch_size=128)
        pred_Y = np.argmax(pred_scores, axis=1)
        create_confusion_matrix(actual=one_hot2indices(test_Y),
                                predicted=pred_Y,
                                class_names=categories,
                                normalise=False)


if __name__ == '__main__':
    main()
