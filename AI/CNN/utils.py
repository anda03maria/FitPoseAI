import numpy as np
import matplotlib.pyplot as plt

def plot_histogram_data(output_data, outputs_name, title):
    plt.hist(output_data, bins=len(outputs_name), edgecolor='black')
    plt.title('Histogram of ' + title)
    plt.xticks(np.arange(len(outputs_name)), outputs_name)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.show()

def plot_confusion_matrix(cm, class_names, title):
    import itertools

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[row, column] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
