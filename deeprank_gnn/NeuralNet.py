import numpy as np
from time import time
import os
import logging
import csv
from typing import List

# torch import
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score

# deeprank_gnn import
from .models.metrics import MetricsExporterCollection
from .DataSet import HDF5DataSet, DivideDataSet, PreCluster
from .Metrics import Metrics


_log = logging.getLogger(__name__)


class NeuralNet(object):

    def __init__(self, database, Net,
                 node_feature=['type', 'polarity', 'bsa'],
                 edge_feature=['dist'], target='irmsd',
                 lr=0.01, weight_decay=0.0000,
                 batch_size=32, percent=[1.0, 0.0],
                 database_eval=None, index=None, class_weights=None, task=None,
                 classes=[0, 1], threshold=None,
                 pretrained_model=None, shuffle=True, outdir='./', cluster_nodes='mcl', transform_sigmoid=False,
                 metrics_exporters: MetricsExporterCollection = None):
        """Class from which the network is trained, evaluated and tested

        Args:
            database (str, required): path(s) to hdf5 dataset(s). Unique hdf5 file or list of hdf5 files.
            Net (function, required): neural network function (ex. GINet, Foutnet etc.)
            node_feature (list, optional): type, charge, polarity, bsa (buried surface area), pssm,
                    cons (pssm conservation information), ic (pssm information content), depth,
                    hse (half sphere exposure).
                    Defaults to ['type', 'polarity', 'bsa'].
            edge_feature (list, optional): dist (distance). Defaults to ['dist'].
            target (str, optional): irmsd, lrmsd, fnat, capri_class, bin_class, dockQ.
                    Defaults to 'irmsd'.
            lr (float, optional): learning rate. Defaults to 0.01.
            weight_decay (float, optional): weight decay, defaults to 0.0000
            batch_size (int, optional): defaults to 32.
            percent (list, optional): divides the input dataset into a training and an evaluation set.
                    Defaults to [1.0, 0.0].
            database_eval ([type], optional): independent evaluation set. Defaults to None.
            index ([type], optional): index of the molecules to consider. Defaults to None.
            class_weights ([list or bool], optional): weights provided to the cross entropy loss function.
                    The user can either input a list of weights or let DeepRanl-GNN (True) define weights
                    based on the dataset content. Defaults to None.
            task (str, optional): 'reg' for regression or 'class' for classification . Defaults to None.
            classes (list, optional): define the dataset target classes in classification mode. Defaults to [0, 1].
            threshold (int, optional): threshold to compute binary classification metrics. Defaults to 4.0.
            pretrained_model (str, optional): path to pre-trained model. Defaults to None.
            shuffle (bool, optional): shuffle the training set. Defaults to True.
            outdir (str, optional): output directory. Defaults to ./
            cluster_nodes (bool, optional): perform node clustering ('mcl' or 'louvain' algorithm). Default to 'mcl'.
        """
        # load the input data or a pretrained model
        # each named arguments is stored in a member vairable
        # i.e. self.node_feature = node_feature
        if pretrained_model is None:
            for k, v in dict(locals()).items():
                if k not in ['self', 'database', 'Net', 'database_eval']:
                    self.__setattr__(k, v)

            if self.task == None:
                if self.target in ['irmsd', 'lrmsd', 'fnat', 'dockQ']:
                    self.task = 'reg'
                elif self.target in ['bin_class', 'capri_classes']:
                    self.task = 'class'
                else:
                    raise ValueError(
                        f"User target detected -> The task argument is required ('class' or 'reg'). \n\t"
                        f"Example: \n\t"
                        f""
                        f"model = NeuralNet(database, GINet,"
                        f"                  target='physiological_assembly',"
                        f"                  task='class',"
                        f"                  shuffle=True,"
                        f"                  percent=[0.8, 0.2])")

            if self.task == 'class' and self.threshold == None:
                print('the threshold for accuracy computation is set to {}'.format(self.classes[1]))
                self.threshold = self.classes[1]
            if self.task == 'reg' and self.threshold == None:
                print('the threshold for accuracy computation is set to 0.3')
                self.threshold = 0.3
            self.load_model(database, Net, database_eval)

        else:
            self.load_params(pretrained_model)
            self.outdir = outdir
            self.load_pretrained_model(database, Net)

        if metrics_exporters is not None:
            self._metrics_exporters = metrics_exporters
        else:
            self._metrics_exporters = MetricsExporterCollection()

    def load_pretrained_model(self, database, Net):
        """
        Loads pretrained model

        Args:
            database (str): path to hdf5 file(s)
            Net (function): neural network
        """
        # Load the test set
        test_dataset = HDF5DataSet(root='./', database=database,
                                   node_feature=self.node_feature, edge_feature=self.edge_feature,
                                   target=self.target, clustering_method=self.cluster_nodes)
        self.test_loader = DataLoader(
            test_dataset)
        PreCluster(test_dataset, method=self.cluster_nodes)

        print('Test set loaded')
        self.put_model_to_device(test_dataset, Net)

        self.set_loss()

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)

        # load the model and the optimizer state if we have one
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def load_model(self, database, Net, database_eval):
        """
        Loads model

        Args:
            database (str): path to the hdf5 file(s) of the training set
            Net (function): neural network
            database_eval (str): path to the hdf5 file(s) of the evaluation set

        Raises:
            ValueError: Invalid node clustering method.
        """
        # dataset
        dataset = HDF5DataSet(root='./', database=database, index=self.index,
                              node_feature=self.node_feature, edge_feature=self.edge_feature,
                              target=self.target, clustering_method=self.cluster_nodes)

        if self.cluster_nodes != None:
            if self.cluster_nodes == 'mcl' or self.cluster_nodes == 'louvain':
                print("Loading clusters")
                PreCluster(dataset, method=self.cluster_nodes)
            else:
                raise ValueError(
                    f"Invalid node clustering method. \n\t"
                    f"Please set cluster_nodes to 'mcl', 'louvain' or None. Default to 'mcl' \n\t")

        # divide the dataset
        train_dataset, valid_dataset = DivideDataSet(
            dataset, percent=self.percent)

        # dataloader
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        print('Training set loaded')

        if self.percent[1] > 0.0:
            self.valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            print('Evaluation set loaded')

        # independent validation dataset
        if database_eval is not None:
            print('Loading independent evaluation dataset...')
            valid_dataset = HDF5DataSet(root='./', database=database_eval, index=self.index,
                                        node_feature=self.node_feature, edge_feature=self.edge_feature,
                                        target=self.target, clustering_method=self.cluster_nodes)

            if self.cluster_nodes == 'mcl' or self.cluster_nodes == 'louvain':
                print('Loading clusters for the evaluation set.')
                PreCluster(valid_dataset, method=self.cluster_nodes)

            self.valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            print('Independent validation set loaded !')

        else:
            print('No independent validation set loaded')

        self.put_model_to_device(dataset, Net)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)

        self.set_loss()

    def put_model_to_device(self, dataset, Net):
        """
        Puts the model on the available device

        Args:
            dataset (str): path to the hdf5 file(s)
            Net (function): neural network

        Raises:
            ValueError: Incorrect output shape
        """
        # get the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        _log.info(f"device set to : {self.device}")
        if self.device.type == 'cuda':
            _log.info(f"cuda device name is {torch.cuda.get_device_name(0)}")

        self.num_edge_features = len(self.edge_feature)

        # regression mode
        if self.task == 'reg':
            self.model = Net(dataset.get(
                0).num_features, 1, self.num_edge_features).to(self.device)

        # classification mode
        elif self.task == 'class':
            self.classes_to_idx = {i: idx for idx,
                                   i in enumerate(self.classes)}
            self.idx_to_classes = {idx: i for idx,
                                   i in enumerate(self.classes)}
            self.output_shape = len(self.classes)

            self.model = Net(dataset.get(
                0).num_features, self.output_shape, self.num_edge_features).to(self.device)

    def set_loss(self):
        """Sets the loss function (MSE loss for regression/ CrossEntropy loss for classification)."""
        if self.task == 'reg':
            self.loss = MSELoss()

        elif self.task == 'class':

            # assign weights to each class in case of unbalanced dataset
            self.weights = None
            if self.class_weights == True:
                targets_all = []
                for batch in self.train_loader:
                    targets_all.append(batch.y)

                targets_all = torch.cat(
                    targets_all).squeeze().tolist()
                self.weights = torch.tensor(
                    [targets_all.count(i) for i in self.classes], dtype=torch.float32)
                print('class occurences: {}'.format(self.weights))
                self.weights = 1.0 / self.weights
                self.weights = self.weights / self.weights.sum()
                print('class weights: {}'.format(self.weights))

            self.loss = nn.CrossEntropyLoss(
                weight=self.weights, reduction='mean')

    def train(self, nepoch=1, validate=False, save_model='last',
              hdf5='train_data.hdf5',
              tensorboard_directory=None):
        """
        Trains the model

        Args:
            nepoch (int, optional): number of epochs. Defaults to 1.
            validate (bool, optional): perform validation. Defaults to False.
            save_model (last, best, optional): save the model. Defaults to 'last'
            hdf5 (str, optional): hdf5 output file
            tensorboard_directory (directory path, optional): where to store the tensorflow files
        """

        train_losses = []
        valid_losses = []

        with self._metrics_exporters:

            # Output file name
            data_hdf5_path = self.update_name(hdf5, self.outdir)

            # Number of epochs
            self.nepoch = nepoch

            self.eval(self.train_loader, 0, "training")
            if validate:
                self.eval(self.valid_loader, 0, "validation")

            # Loop over epochs
            self.data = {}
            for epoch in range(1, nepoch + 1):

                # Train the model
                self.model.train()

                loss_ = self._epoch(epoch, "training")

                train_losses.append(loss_)


                # Validate the model
                if validate:
                    t0 = time()
                    loss_ = self.eval(self.valid_loader, epoch, "validation")
                    t = time() - t0

                    valid_losses.append(loss_)

                    # save the best model (i.e. lowest loss value on validation data)
                    if save_model == 'best':

                        if min(valid_losses) == _val_loss:
                            self.save_model(filename='t{}_y{}_b{}_e{}_lr{}_{}.pth.tar'.format(
                                self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr), str(epoch)))

                else:
                    # if no validation set, saves the best performing model on the traing set
                    if save_model == 'best':
                        if min(train_losses) == _loss:
                            _log.warning("""The training set is used both for learning and model selection.
                                            This may lead to training set data overfitting.
                                            We advice you to use an external validation set.""")

                            self.save_model(filename='t{}_y{}_b{}_e{}_lr{}_{}.pth.tar'.format(
                                self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr), str(epoch)))

            # Save the last model
            if save_model == 'last':
                self.save_model(filename='t{}_y{}_b{}_e{}_lr{}.pth.tar'.format(
                    self.task, self.target, str(self.batch_size), str(nepoch), str(self.lr)))


    def test(self, database_test=None, threshold=4, hdf5='test_data.hdf5',
             tensorboard_directory=None):
        """
        Tests the model

        Args:
            database_test ([type], optional): test database
            threshold (int, optional): threshold use to tranform data into binary values. Defaults to 4.
            hdf5 (str, optional): output hdf5 file. Defaults to 'test_data.hdf5'.
            tensorboard_directory (directory path, optional): where to store the tensorflow files
        """

        with self._metrics_exporters:

            # Output file name
            data_hdf5_path = self.update_name(hdf5, self.outdir)

            # Loads the test dataset if provided
            if database_test is not None:

                # Load the test set
                test_dataset = HDF5DataSet(root='./', database=database_test,
                                           node_feature=self.node_feature, edge_feature=self.edge_feature,
                                           target=self.target, clustering_method=self.cluster_nodes)

                PreCluster(test_dataset, method='mcl')

                self.test_loader = DataLoader(test_dataset)

            else:
                if self.load_pretrained_model == None:
                    raise ValueError(
                        "You need to upload a test dataset \n\t"
                        "\n\t"
                        ">> model.test(test_dataset)\n\t"
                        "if a pretrained network is loaded, you can directly test the model on the loaded dataset :\n\t"
                        ">> model = NeuralNet(database_test, gnn, pretrained_model = model_saved, target=None)\n\t"
                        ">> model.test()\n\t")
            self.data = {}

            # Run test
            loss_ = self.eval(self.test_loader, 0, "testing")

    def eval(self, loader: DataLoader, epoch_number: int, pass_name: str) -> float:
        """
        Evaluates the model

        Args:
            loader: data to evaluate on
            epoch_number: number for this epoch, used for storing the metrics
            pass_name: 'training', 'validation' or 'testing'

        Returns:
            running loss:
        """
        self.model.eval()

        loss_func = self.loss

        targets = []
        outputs = []
        entry_names = []

        batch_count = len(loader)
        sum_of_losses = 0
        count_predictions = 0

        t0 = time()
        for batch_index, data_batch in enumerate(loader):

            data_batch = data_batch.to(self.device)
            pred = self.model(data_batch)
            pred, data_batch.y = self.format_output(
                pred, data_batch.y)

            # Check if a target value was provided (i.e. benchmarck scenario)
            if data_batch.y is not None:
                targets += data_batch.y.tolist()
                loss_ = loss_func(pred, data_batch.y)

                count_predictions += pred.shape[0]
                sum_of_losses += loss_.detach().item() * pred.shape[0]

            # get the outputs for export
            if self.task == 'class':
                pred = F.softmax(pred.detach(), dim=1)
            else:
                pred = pred.detach().reshape(-1)

            outputs += pred.tolist()

            # get the data
            entry_names += data_batch['mol']

        dt = time() - t0

        if count_predictions > 0:
            eval_loss = sum_of_losses / count_predictions
        else:
            eval_loss = 0.0

        self._metrics_exporters.process(pass_name, epoch_number, entry_names, outputs, targets)
        self.log_epoch_data(pass_name, epoch_number, eval_loss, dt)

        return eval_loss

    def _epoch(self, epoch_number: int, pass_name: str) -> float:
        """
        Runs a single epoch

        Args:
            epoch_number (int)
            pass_name (str): 'training', 'validation' or 'testing'

        Returns:
            running loss
        """

        sum_of_losses = 0
        count_predictions = 0

        targets = []
        outputs = []
        entry_names = []

        batch_count = len(self.train_loader)

        t0 = time()
        for batch_index, data_batch in enumerate(self.train_loader):

            data_batch = data_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data_batch)
            pred, data_batch.y = self.format_output(pred, data_batch.y)

            loss_ = self.loss(pred, data_batch.y)
            loss_.backward()
            self.optimizer.step()

            count_predictions += pred.shape[0]
            sum_of_losses += loss_.detach().item() * pred.shape[0]  # convert mean back to sum

            targets += data_batch.y.tolist()

            # get the outputs for export
            if self.task == 'class':
                pred = F.softmax(pred.detach(), dim=1)
            else:
                pred = pred.detach().reshape(-1)

            outputs += pred.tolist()

            # get the data
            entry_names += data_batch['mol']

        dt = time() - t0

        if count_predictions > 0:
            epoch_loss = sum_of_losses / count_predictions
        else:
            epoch_loss = 0.0

        self._metrics_exporters.process(pass_name, epoch_number, entry_names, outputs, targets)
        self.log_epoch_data(pass_name, epoch_number, epoch_loss, dt)

        return epoch_loss

    def compute_class_weights(self):

        targets_all = []
        for batch in self.train_loader:
            targets_all.append(batch.y)

        targets_all = torch.cat(targets_all).squeeze().tolist()
        weights = torch.tensor([targets_all.count(i)
                                for i in self.classes], dtype=torch.float32)
        print('class occurences: {}'.format(weights))
        weights = 1.0 / weights
        weights = weights / weights.sum()
        print('class weights: {}'.format(weights))
        return weights

    @staticmethod
    def log_epoch_data(stage, epoch, loss, time):
        """
        Prints the data of each epoch

        Args:
            stage (str): train or valid
            epoch (int): epoch number
            loss (float): loss during that epoch
            time (float): timing of the epoch
        """

        _log.info('Epoch [%04d] : %s loss %e | time %1.2e sec.' % (epoch, stage, loss, time))

    def format_output(self, pred, target=None):
        """Format the network output depending on the task (classification/regression)."""
        if self.task == 'class' :
            #pred = F.softmax(pred, dim=1)

            if target is not None:
                target = torch.tensor(
                    [self.classes_to_idx[int(x)] for x in target]).to(self.device)

        elif self.transform_sigmoid is True :
            pred = torch.sigmoid(pred.reshape(-1))

        else:
            pred = pred.reshape(-1)

        return pred, target

    @staticmethod
    def update_name(hdf5, outdir):
        """
        Checks if the file already exists, if so, update the name

        Args:
            hdf5 (str): hdf5 file
            outdir (str): output directory

        Returns:
            str: update hdf5 name
        """
        fname = os.path.join(outdir, hdf5)

        count = 0
        hdf5_name = hdf5.split('.')[0]

        # If file exists, change its name with a number
        while os.path.exists(fname):
            count += 1
            hdf5 = '{}_{:03d}.hdf5'.format(hdf5_name, count)
            fname = os.path.join(outdir, hdf5)

        return fname

    def save_model(self, filename='model.pth.tar'):
        """
        Saves the model to a file

        Args:
            filename (str, optional): name of the file. Defaults to 'model.pth.tar'.
        """
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'node': self.node_feature,
                 'edge': self.edge_feature,
                 'target': self.target,
                 'task': self.task,
                 'classes': self.classes,
                 'class_weight': self.class_weights,
                 'batch_size': self.batch_size,
                 'percent': self.percent,
                 'lr': self.lr,
                 'weight_decay': self.weight_decay,
                 'index': self.index,
                 'shuffle': self.shuffle,
                 'threshold': self.threshold,
                 'cluster_nodes': self.cluster_nodes,
                 'transform_sigmoid': self.transform_sigmoid}

        torch.save(state, filename)

    def load_params(self, filename):
        """
        Loads the parameters of a pretrained model

        Args:
            filename ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(filename, map_location=torch.device(self.device))

        self.node_feature = state['node']
        self.edge_feature = state['edge']
        self.target = state['target']
        self.batch_size = state['batch_size']
        self.percent = state['percent']
        self.lr = state['lr']
        self.weight_decay = state['weight_decay']
        self.index = state['index']
        self.class_weights = state['class_weight']
        self.task = state['task']
        self.classes = state['classes']
        self.threshold = state['threshold']
        self.shuffle = state['shuffle']
        self.cluster_nodes = state['cluster_nodes']
        self.transform_sigmoid = state['transform_sigmoid']

        self.opt_loaded_state_dict = state['optimizer']
        self.model_load_state_dict = state['model']

