import torch
import copy
import logging
import numpy as np
import torch
import torch.optim as optim
import time
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from base.base_net import BaseNet
from base.base_trainer import BaseTrainer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve
from utils.write2txt import writer2txt
from collections import Counter

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

class GADFTrainer(BaseTrainer):

    def __init__(self, c, anchor, eta_0: float, eta_1: float, eta_2: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, sample_count: int=100, debug=False):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.sample_count = sample_count
        self.debug = debug

        # parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.anchor = torch.tensor(anchor, device=self.device) if anchor is not None else None

        self.eta = torch.tensor(1, device=self.device)
        self.eta_0 = torch.tensor(eta_0, device=self.device)
        self.eta_1 = torch.tensor(eta_1, device=self.device) if eta_1 is not None else None
        self.eta_2 = torch.tensor(eta_2, device=self.device) if eta_2 is not None else None

        # Optimization parameters
        self.eps = torch.tensor(1e-6, device=self.device)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            self.c = self.init_center_c(train_loader, net)

        self.has_non_target_flag = (dataset.train_set.semi_targets == -2).sum() != 0
        if self.has_non_target_flag == False:
            logger.error("The input must contain non-target exceptions")

        # Initialize anchor (if anchor not loaded)
        if self.anchor is None:
            self.anchor = self.init_and_update_center_anchor(train_loader, net)

        if self.debug:
            list_hard_sample_count = []
            list_easy_sample_count = []
            list_hard_sample_sum = []
            list_easy_sample_sum = []

        # Training
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):
            scheduler.step()
            
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            

            self.anchor = self.init_and_update_center_anchor(train_loader, net)
            
            writer = writer2txt()
            writer.log("distance_c_to_anchor: {:.6f}".format(torch.sum((self.c - self.anchor) ** 2)))

            if self.debug:
                hard_sample_count = 0
                hard_target_loss_sum = 0.0
                easy_sample_count = 0
                easy_target_loss_sum = 0.0

            for data in train_loader:
                inputs, _, semi_targets = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                
                # Calculate the distance between c and anchor.
                distance_c_anchor = torch.sum((self.c - self.anchor) ** 2)
                dist_to_c = torch.sum((outputs - self.c) ** 2, dim=1)
                dist_to_anchor = torch.sum((outputs - self.anchor) ** 2, dim=1)
                
                if self.debug:
                    # If the distance is greater than or equal to distance_c_anchor, set mask_easy to 1; otherwise, set mask_easy to 0.
                    # mask_hard is the opposite of mask_easy.
                    # These two items only take effect when semi=-1.
                    mask_easy = torch.where(dist_to_c >= distance_c_anchor, torch.tensor([1.0]).cuda(), torch.tensor([0.0]).cuda())
                    mask_hard = torch.where(dist_to_c < distance_c_anchor, torch.tensor([1.0]).cuda(), torch.tensor([0.0]).cuda())
                    
                    # target_to_c refers to the loss between target anomalies and c.
                    # target_to_anchor refers to the loss between target anomalies and anchor.
                    target_to_c = ((dist_to_c + self.eps) ** semi_targets.float())
                    target_to_anchor = ((dist_to_anchor + self.eps) ** semi_targets.float())
                    
                    # calculate the loss of easy-target
                    easy_target_loss = self.eta_1 * (target_to_c * mask_easy + target_to_anchor * mask_easy)
                    # calculate the loss of hard-target
                    hard_target_loss = self.eta_2 * target_to_c * mask_hard
                    # the loss of all target-anomalies
                    target_loss = easy_target_loss + hard_target_loss

                    hard_sample_count += torch.where(semi_targets == -1, mask_hard, torch.tensor([0.0]).cuda()).sum().item()
                    hard_target_loss_sum += torch.where(semi_targets == -1, hard_target_loss, torch.zeros_like(hard_target_loss)).sum().item()

                    easy_sample_count += torch.where(semi_targets == -1, mask_easy, torch.tensor([0.0]).cuda()).sum().item()
                    easy_target_loss_sum += torch.where(semi_targets == -1, easy_target_loss, torch.zeros_like(easy_target_loss)).sum().item()
                    
                    normal_and_non_target_loss = torch.where(semi_targets == -2, self.eta_0 * dist_to_c, dist_to_c)
                    
                    losses = torch.where(semi_targets == -1, target_loss, normal_and_non_target_loss)
                    loss = torch.mean(losses)
                
                else:
                    nomral_weights = (semi_targets == 0)
                    easy_weights = self.eta_1 * ((semi_targets == -1) & (dist_to_c >= distance_c_anchor))
                    hard_weights = self.eta_2 * ((semi_targets == -1) & (dist_to_c < distance_c_anchor))
                    anchor_weights = self.eta_0 * ((semi_targets == -2))
                    weights = nomral_weights + easy_weights + hard_weights + anchor_weights
                    loss = torch.mean(weights.float() * torch.where(semi_targets != -1, dist_to_c, 1 / (dist_to_c + self.eps)) + easy_weights.float() * 1 / (dist_to_anchor + self.eps))


                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s 'f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            # print('hard_sample_count: {:.2f} | hard_sample_loss_sum: {:.2f} | easy_sample_count: {:.2f} | easy_sample_loss_sum: {:.2f}'.format(hard_sample_count, hard_target_loss_sum, easy_sample_count, easy_target_loss_sum))
            
            if self.debug:
                writer.log(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s 'f'| Train Loss: {epoch_loss / n_batches:.6f} |' + 'hard_sample_count: {:.2f} | hard_sample_loss_sum: {:.2f} | easy_sample_count: {:.2f} | easy_sample_loss_sum: {:.2f}'.format(hard_sample_count, hard_target_loss_sum, easy_sample_count, easy_target_loss_sum))

                list_hard_sample_count.append(hard_sample_count)
                list_easy_sample_count.append(easy_sample_count)
                list_hard_sample_sum.append(hard_target_loss_sum)
                list_easy_sample_sum.append(easy_target_loss_sum)

        self.train_time = time.time() - start_time

        if self.debug:
            print('list_hard_sample_count', list_hard_sample_count)
            print('list_easy_sample_count', list_easy_sample_count)
            print('list_hard_sample_sum', list_hard_sample_sum)
            print('list_easy_sample_sum', list_easy_sample_sum)
            list_hard_sample_count = np.array(list_hard_sample_count)
            list_easy_sample_count = np.array(list_easy_sample_count)
            list_hard_sample_sum = np.array(list_hard_sample_sum)
            list_easy_sample_sum = np.array(list_easy_sample_sum)

        return net

    def test(self, dataset, net: BaseNet):
        logger = logging.getLogger()

        writer = writer2txt()
        
        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        inputs_list = np.empty((0,1,28,28))
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)
        precision, recall, threshold = precision_recall_curve(labels, scores)
        self.test_auc_pr = auc(recall, precision)
        # plt.plot(recall, precision, marker='.', label='---')
        # plt.cla()
        # plt.plot(recall, precision)
        # plt.savefig("1.pdf",format="pdf")
        self.f1 = np.nanmax((2 * precision * recall) / (precision + recall))
        print(self.f1)
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

        test_out = np.int64(scores >= optimal_th)
        cm = confusion_matrix(labels, test_out, labels=[0, 1])
        # print(cm)
        # print(cm[0][0],cm[0][1],cm[1][0],cm[1][1],self.test_auc,self.test_auc_pr)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        # print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        # print('Test AUC-PR: {:.2f}%'.format(100. * self.test_auc_pr))
        # print('Test AUC: {:.2f}% | Test PRC: {:.2f}%'.format(100. * self.test_auc, 100. * self.test_auc_pr))
        print('Test AUC: {:.2f}% | Test PRC: {:.2f}% | Test F1: {:.2f}'.format(100. * self.test_auc, 100. * self.test_auc_pr, self.f1 * 100))
        writer.log('Test AUC: {:.2f}% | Test PRC: {:.2f}% | Test F1: {:.2f}'.format(100. * self.test_auc, 100. * self.test_auc_pr, self.f1 * 100))


        # print('Test Time: {:.3f}s'.format(self.`test_time))
        # print('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def init_and_update_center_anchor(self, train_loader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center anchor as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        # zero = torch.zeros(self.batch_size, device=self.device)

        net.eval()
        with torch.no_grad():
            inputs = train_loader.dataset.data[np.where(train_loader.dataset.semi_targets == -2)[0]]
            if isinstance(inputs, np.ndarray):
                inputs = torch.Tensor(inputs)
                inputs = inputs.to(self.device)
            else:
                inputs = inputs.float().to(self.device)
            outputs = net(inputs)
            c = torch.sum(outputs, dim=0)
            c /= outputs.shape[0]
        
            # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

        return c