import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

def plot_loss(loss, val_loss):
    # Save Train result
    fig = plt.gcf()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot_accuracy(accuracy, val_accuracy):
    # Save Train result
    fig = plt.gcf()
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def confusion_matrix(y_true, y_pred):
    container = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()