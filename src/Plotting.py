import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold
import numpy as np

class Plotting(object):

    @staticmethod
    def plot_performance(performances_1,performances_2):
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(performances_1)), performances_1, 'b',np.arange(len(performances_2)), performances_2, 'g')


    @staticmethod
    def plot_embedding(features, classes, labels, title=None):
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        features = (features - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(features.shape[0]):
            plt.text(features[i, 0], features[i, 1], str(labels[i]),
                     color=plt.cm.Set1(float(classes[i] / 60)),
                     fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(features.shape[0]):
                dist = np.sum((features[i] - shown_images) ** 2, 1)
                # if np.min(dist) < 4e-3:
                # don't show points that are too close
                #    continue
                shown_images = np.r_[shown_images, [features[i]]]
                """imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)"""
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    @staticmethod
    def plot_distribution_t_SNE(vectors, colors, labels,title="T-SNE Embedding"):
        print("Computing t-SNE embedding")

        x = np.asarray(vectors)
        # x = preprocessing.normalize(x, norm='l2')

        tsne = manifold.TSNE(n_components=2, init='pca', perplexity=2, n_iter=20000, early_exaggeration=4,
                             learning_rate=100, method="exact")
        X_tsne = tsne.fit_transform(x)

        Plotting.plot_embedding(X_tsne, np.asarray(colors), labels,
                       title)


