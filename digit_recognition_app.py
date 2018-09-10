import argparse
import cv2
from keras.models import load_model
from digit_recognition import pre_processing


def main(image_path, invert=False, show=False, scores=False, thresh=5):
    # Load a saved model
    model = load_model('digit.keras')
    image = cv2.imread(image_path)
    processed_image, x = pre_processing(image, invert=invert, thresh=thresh)

    # Show the image
    if show:
        cv2.imshow('processed_image', processed_image)
        cv2.waitKey(0)  # Wait for a keypress
        cv2.destroyAllWindows()

    # Get the prediction
    X = x.reshape((1, x.shape[0]))  # Reshape into a matrix
    Y = model.predict(x=X)
    y = Y[0]  # Get result from matrix

    # Show the scores for each category
    if scores:
        for i in range(len(y)):
            score_pc = round(y[i] * 100, 1)
            print(i, ': ', score_pc, '%')

    # Show the top prediction
    print('\nPrediction: ', y.argmax())


if __name__ == '__main__':
    # Construct input parameter parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to the image')
    parser.add_argument('--show', default=False, help='Show the image. True/False')
    parser.add_argument('--invert', default=False, help='Invert the image. True/False')
    parser.add_argument('--scores', default=False, help='Show the scores. True/False')
    parser.add_argument('--thresh', default=False, help='Threshold for binary image. 0-255. "None" to run off.')
    args = parser.parse_args()

    # Eval inputs from string
    args.thresh = eval(args.thresh)

    # Run the main code
    main(image_path=args.image, invert=args.invert, show=args.show, scores=args.scores, thresh=args.thresh)
