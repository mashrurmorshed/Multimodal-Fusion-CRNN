from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def make_confusion_matrix(y_true, y_pred, out_path):
    CM = confusion_matrix(y_true, y_pred)
    CM = CM / CM.sum(axis=1) * 100

    G = ["G", "T", "E", "P", "R-CW", "R-CCW",
         "S-R", "S-L", "S-U", "S-D", "S-X",
         "S-V", "S-+", "Sh"]

    plt.figure(figsize=(10, 10))

    sns.heatmap(CM, annot=True, annot_kws={"size": 11},  fmt=".1f", cmap="Blues", cbar=False,
            xticklabels=G, yticklabels=G)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    plt.savefig(out_path)
    plt.show()