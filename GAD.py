from optim.GADSTrainer import GADSTrainer
from optim.GADSTrainer import GADSTrainer
from networks.main import build_network, build_autoencoder
from optim.AETrainer import AETrainer
import numpy as np
import json
import torch
from utils.write2txt import writer2txt
from collections import Counter

class GAD(object):

    def __init__(self, eta_0: float = 1.0, eta_1: float = 1.0, eta_2: float = 2.0, model_type: str = "BiasedAD", update_anchor=None, debug=False, update_epoch = None):
        """Inits GAD with hyperparameter eta."""
        self.eta_0 = eta_0
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.c = None  # hypersphere center c
        self.anchor = None 
        self.model_type = model_type
        self.update_anchor = update_anchor
        self.debug = debug
        self.update_epoch = update_epoch

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_auc_pr': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, sample_count: int=100):
        """Trains the GAD model on the training data."""

        self.optimizer_name = optimizer_name
        
        if self.model_type == "BiasedAD":
            self.trainer = GADSTrainer(self.c, self.anchor, self.eta_0, self.eta_1, self.eta_2, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                    lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                    device=device, n_jobs_dataloader=n_jobs_dataloader, sample_count=sample_count, debug = self.debug)
        elif self.model_type == "BiasedADM":
            self.trainer = GADSTrainer(self.c, self.anchor, self.eta_0, self.eta_1, self.eta_2, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                    lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                    device=device, n_jobs_dataloader=n_jobs_dataloader, sample_count=sample_count, update_anchor=self.update_anchor, debug = self.debug, update_epoch = self.update_epoch)
        # run time statistics

        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp.add_function(self.trainer.init_and_update_center_anchor)
        # lp_wrapper  = lp(self.trainer.train)
        # self.net = lp_wrapper(dataset, self.net)
        # lp.print_stats()

        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list
        self.anchor = self.trainer.anchor.cpu().data.numpy().tolist()

    def test(self, dataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the GAD model on the test data."""

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_auc_pr'] = self.trainer.test_auc_pr
        self.results['test_scores'] = self.trainer.test_scores
        self.results['test_f1'] = self.trainer.f1
        
        writer2txt().write_txt('Test AUC: {:.2f}% | Test PRC: {:.2f}% | Test F1: {:.2f}'.format(100. * self.results['test_auc'], 100. * self.results['test_auc_pr'], self.results['test_f1'] * 100))

    def pretrain(self, dataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the GAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize GAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the GAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save GAD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load GAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def intermediate_result(self, data_loader, device: str = 'cuda'):
        net = self.net
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            intermediate_data_input = []
            intermediate_data_label = []
            intermediate_data_semi_target = []

            for data in data_loader:
                if len(data) == 5:
                    idx, inputs, labels, semi_targets, _ = data
                if len(data) == 3:
                    inputs, labels, semi_targets = data
                inputs = inputs.to(device)
                outputs = net(inputs)
                
                intermediate_data_input.append(outputs.cpu().numpy())
                intermediate_data_label.append(labels.numpy())
                intermediate_data_semi_target.append(semi_targets.numpy())

            intermediate_data_input = np.concatenate(intermediate_data_input)
            intermediate_data_label = np.concatenate(intermediate_data_label)
            intermediate_data_semi_target = np.concatenate(intermediate_data_semi_target)

        return intermediate_data_input, intermediate_data_label, intermediate_data_semi_target