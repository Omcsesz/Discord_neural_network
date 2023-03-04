
import logging
from keras.models import Sequential
from keras.layers import Dense
from csv_processing import process

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    try:
        X_train, X_test, y_train, y_test = process()
        _logger.info(X_train.shape[1])
        model = Sequential()
        model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(7, activation='elu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=20, batch_size=133, verbose=1)
        y_pred = model.predict(X_test)
        #for i in range(100):
        #    print(y_pred[i])
        score = model.evaluate(X_test, y_test, verbose=1)
        _logger.info(score)
        return 0
    except Exception as e:
        _logger.error(e)
        return 1

if __name__ == "__main__":
    exit(main())
