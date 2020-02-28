import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
#from sklearn import manifold


class TensorboardXLogger:
    def __init__(self, path, name=''):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.writer = SummaryWriter(path+"/"+name)
        self.iteration = 0
        self.name = name
        self.path = path

    def per_batch_results(self, acc_dict):
        file = open(self.path + "/results.txt", "a")
        for mname in acc_dict:
            results = ",".join([f"{acc_dict[mname]:.3f}"])
            file.write(f"{self.name},{mname},{results}\n")
        file.close()

    def write_results(self, name, cumulative_accuracies, iteration, type_results):
        file = open(self.path+f"/{name}.txt", "a")
        file.write(f'\n\nType results: {type_results} | Iteration: {iteration}')

        for type, res in cumulative_accuracies.items():
            #type could be acc_cum, new, base
            file.write(f'\n\nType evaluation: {type}\n')
            # ADD VALUES
            for metrics, values in res.items():
                file.write(f"\n{metrics} " + " ".join([str(f' {x:.3f} ') for x in values]))

        file.close()

    def save_results(self, name, acc_base, acc_new, acc_cum, iteration, type_results):
        # type_results+'/results/' + name + '_base' per avere i 3 plot diversi per ogni metodo
        self.writer.add_scalar(type_results + '/results/base', acc_base['nme'], iteration)
        self.writer.add_scalar(type_results + '/results/new', acc_new['nme'], iteration)
        self.writer.add_scalar(type_results + '/results/cum', acc_cum['nme'], iteration)

    def save_cumulative_results(self, results):
        # CUMULATIVE RESULTS DICTIONARY
        text = ''
        for type, res in results.items():
            # CREATE TABLE
            text += f'<table width=\"100%\"><td>{type}</td>'

            # ADD THE CORRECT NUMBER OF COLUMN ACCORDING TO THE NUMBER OF BATCHES
            for index in range(len(res['nme'])):
                text += f'<td>{index}</td>'

            # ADD VALUES
            for metrics, values in res.items():
                text += f"<tr><td>{metrics}</td>" + " ".join([str(f'<td>{x:.3f}</td>') for x in values]) + "</tr>"
            text += "</table>"
        self.writer.add_text('results', text, 0)

    def track_means(self, class_means, sqd, all_features_extracted, iteration):
        # self.writer.add_histogram(f'class_means/{self.name}', class_means, global_step=self.iteration)
        np.save(self.path + f"/class_means@iteration-{iteration}", class_means)
        np.save(self.path + f"/sqd@iteration-{iteration}", sqd)
        np.save(self.path + f"/extracted_feature@iteration-{iteration}", all_features_extracted)

    def print_accuracy(self, method, acc_base, acc_new, acc_cum):
        logging.info("Cumulative results")
        logging.info(f" Accuracy NME {method:<15}:\t{acc_cum['nme']:.2f}")
        logging.info(f" Accuracy CNN {method:<15}:\t{acc_cum['cnn']:.2f}")

        logging.info("New batch results")
        logging.info(f" Accuracy NME {method:<15}:\t{acc_new['nme']:.2f}")
        logging.info(f" Accuracy CNN {method:<15}:\t{acc_new['cnn']:.2f}")

        logging.info("First results")
        logging.info(f" Accuracy NME {method:<15}:\t{acc_base['nme']:.2f}")
        logging.info(f" Accuracy CNN {method:<15}:\t{acc_base['cnn']:.2f}")
        logging.info("")

    def log_training(self, epoch, train_loss, train_acc, valid_loss, valid_acc, iteration, **kwargs):
        logging.info(f"Epoch {epoch + 1:3d} : Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
                     f"          : Valid Loss {valid_loss:.6f}, Valid Acc {valid_acc:.2f}")
        if self.iteration != iteration:
            self.iteration = iteration
        # SALVO PER OGNI ITERAZIONE (N CLASSI POI N+M...) LA LOSS E L'ACCURATEZZA SU TRAINING E VALDIDATION
        # ALL'AUMENTARE DELLE EPOCHE
        # self.writer.add_scalar(f'loss/loss-{iteration}', {'train': train_loss, 'valid': valid_loss}, epoch)
        # self.writer.add_scalar(f'acc/acc-{iteration}', {'train': train_acc, 'valid': valid_acc}, epoch)

        self.writer.add_scalar(f'loss/train-{iteration}', train_loss, epoch)
        self.writer.add_scalar(f'loss/valid-{iteration}', valid_loss, epoch)
        # NB: siccome non usiamo più le predizioni non ha senso calcolare l'accuratezza come prima
        # self.writer.add_scalar(f'acc/train-{iteration}', train_acc, epoch)
        # self.writer.add_scalar(f'acc/val-{iteration}', valid_acc, epoch)

        for k in kwargs:
            self.writer.add_scalar(k, kwargs[k], epoch)

    @staticmethod
    def conf_matrix_figure(cm):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig

    def confusion_matrix(self, y, y_hat, n_classes, results_type):
        conf = np.zeros((n_classes, n_classes))

        for i in range(len(y)):
            conf[y[i], y_hat[i]] += 1

        cm = conf.astype('float') / (conf.sum(axis=1) + 0.000001)[:, np.newaxis]

        fig = self.conf_matrix_figure(cm)
        self.writer.add_figure('conf_matrix/'+results_type, fig, global_step=self.iteration, close=True)

        avg_acc = np.diag(cm).mean() * 100.
        print(f"Per class accuracy ({results_type}): {avg_acc}")
        return conf

    @staticmethod
    def tsne_figure(x, y, n_classes):

        number_of_colors = n_classes
        seeds= ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        colors = ["#" + ''.join([np.random.choice(seeds, replace=False) for j in range(6)]) for i in range(number_of_colors)]

        fig, ax = plt.subplots()
        ax.set(title=f't-SNE',
               ylabel='Dimension 2',
               xlabel='Dimension 1')

        for g in np.unique(y):
            ix = np.where(y == g)
            ax.scatter(x[ix, 0], x[ix, 1], c=colors[g], label=g, alpha=0.5)

        # ax.legend()

        return fig


    def tsne(self, x, y, results_type, n_classes):
        res = manifold.TSNE(n_components=2, random_state=0).fit_transform(x)
        fig = self.tsne_figure(res, y, n_classes)
        self.writer.add_figure('t-SNE/'+results_type, fig, global_step=self.iteration, close=True)


    @staticmethod
    def means_matrix_figure(matrix):
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, interpolation='nearest', aspect='auto', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title='Means Matrix',
               ylabel='Features extracted',
               xlabel='Classes')

        fig.tight_layout()
        return fig

    def means_matrix(self, class_means):
        fig = self.means_matrix_figure(class_means)
        self.writer.add_figure('means_matrix/', fig, global_step=self.iteration)

        return class_means