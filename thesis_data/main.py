
import logging
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from csv_processing import process

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    try:
        X_train, X_test, y_train, y_test = process()
        model = Sequential()
        model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=50, batch_size=133, verbose=1)
        y_pred = model.predict(X_test)
        good = 0
        predictions = pd.DataFrame(0, index=range(1000), columns=range(3))
        for i in range(1000):
            predictions[0][i] = y_test[i]
            predictions[1][i] = y_pred[i]
            if abs(y_pred[i] - y_test[i]) < 0.062:
                good += 1
                predictions[2][i] = 1
            else:
                predictions[2][i] = 0
        print(f'Accuracy: {good/1000}')
        score = model.evaluate(X_test, y_test, verbose=1)
        _logger.info(score)
        return 0
    except Exception as e:
        _logger.error(e)
        return 1

if __name__ == "__main__":
    exit(main())
