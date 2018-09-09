import cv2
from keras.models import load_model
from digit_recognition import pre_processing

show = True
image_path = 'test_image.jpg'


# image_path = 'data/mnist_png/training/1/99.png'

def main():
    # Load a saved model
    model = load_model('digit.keras')
    image = cv2.imread(image_path)
    processed_image, x = pre_processing(image, invert=True)

    if show:
        cv2.imshow('processed_image', processed_image)
        cv2.waitKey(0)  # Wait for a keypress
        cv2.destroyAllWindows()

    # Get the prediction
    X = x.reshape((1, x.shape[0]))  # Reshape into a matrix
    Y = model.predict(x=X)
    y = Y[0]  # Get result from matrix

    # Show the scores for each category
    for i in range(len(y)):
        score_pc = round(y[i] * 100, 1)
        print(i, ': ', score_pc, '%')

    # Show the top prediction
    print('\nPrediction: ', y.argmax())


if __name__ == '__main__':
    main()
