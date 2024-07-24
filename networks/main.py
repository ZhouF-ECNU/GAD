from .fmnist_LeNet import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder
from .mlp import MLP, MLP_Autoencoder


def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('fmnist_LeNet', "mlp_for_nb15", "mlp_for_sqb", "artificial")
    assert net_name in implemented_networks

    net = None

    if net_name == 'fmnist_LeNet':
        net = FashionMNIST_LeNet()

    if net_name == 'mlp_for_nb15':
        net = MLP(x_dim=196, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'mlp_for_sqb':
        net = MLP(x_dim=183, h_dims=[128, 64], rep_dim=32, bias=False)

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('fmnist_LeNet', "mlp_for_nb15", "mlp_for_sqb", "artificial")

    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'fmnist_LeNet':
        ae_net = FashionMNIST_LeNet_Autoencoder()

    if net_name == 'mlp_for_nb15':
        ae_net = MLP_Autoencoder(x_dim=196, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'mlp_for_sqb':
        ae_net = MLP_Autoencoder(x_dim=183, h_dims=[128, 64], rep_dim=32, bias=False)

    return ae_net
