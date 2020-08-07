import torch
import torch.nn as nn
import torch.distributions as D
from utils.common import conv_out_dim


class LinearFeatureExtracor(nn.Module):
    def __init__(self, num_inputs, hidden_layers, batch_normalization=False, activation=nn.ReLU()):
        super(LinearFeatureExtracor, self).__init__()

        layers = []
        self.features_space = num_inputs
        for layer_size in hidden_layers:
            if batch_normalization:
                layers += [nn.Linear(self.features_space, layer_size), nn.BatchNorm1d(layer_size), activation]
            else:
                layers += [nn.Linear(self.features_space, layer_size), activation]
            self.features_space = layer_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvNetFeatureExtracor(nn.Module):
    """
    A 2d image feature extractor from convolution layers followed by optional fc layers
    default parameters are the ones used for Atari Breakout
    """
    def __init__(self, input_shape, fe_layers):
        super(ConvNetFeatureExtracor, self).__init__()
        layers = []
        input_channels = input_shape[0]
        out_shape = input_shape
        for t in fe_layers:
            if type(t) == tuple:
                layers += [nn.Conv2d(input_channels, t[0], kernel_size=t[1], stride=t[2]), nn.ReLU()]
                input_channels = t[0]
                out_shape = (t[0], conv_out_dim(out_shape[1], t[1], t[2]), conv_out_dim(out_shape[2], t[1], t[2]))
        self.cnn_out_dim = out_shape[0] * out_shape[1] * out_shape[2]
        self.conv_head = nn.Sequential(*layers)

        self.features_space = self.cnn_out_dim
        layers = []
        for t in fe_layers:
            if type(t) == int:
                layers += [nn.Linear(self.features_space, t), nn.ReLU()]
                self.features_space = t

        self.post_cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_head(x)
        x = x.view(-1, self.cnn_out_dim)
        x = self.post_cnn(x)
        return x


class DiscreteActor(nn.Module):
    def __init__(self, input_space, action_dim, hidden_layers, activation=nn.ReLU()):
        super(DiscreteActor, self).__init__()

        layers = []
        last_features_space = input_space
        for layer_size in hidden_layers:
            layers += [nn.Linear(last_features_space, layer_size), activation]
            last_features_space = layer_size
        layers += [nn.Linear(last_features_space, action_dim), nn.Softmax(dim=1)]
        self.head = nn.Sequential(*layers)

    def get_dist(self, features):
        probs = self.head(features)
        dist = D.Categorical(probs)
        return dist


class CountinousActor(nn.Module):
    def __init__(self, input_space, action_dim, hidden_layers, activation=nn.ReLU()):
        super(CountinousActor, self).__init__()
        self.action_dim = action_dim
        layers = []
        last_features_space = input_space
        for layer_size in hidden_layers:
            layers += [nn.Linear(last_features_space, layer_size), activation]
            last_features_space = layer_size
        layers += [nn.Linear(last_features_space, action_dim), nn.Tanh()]
        self.log_sigma = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        self.head = nn.Sequential(*layers)

    def get_dist(self, features):
        mu = self.head(features)
        dist = D.multivariate_normal.MultivariateNormal(mu, torch.diag_embed(self.log_sigma.exp()))

        return dist


class Critic(nn.Module):
    def __init__(self, input_space, hidden_layers, activation=nn.ReLU()):
        super(Critic, self).__init__()
        layers = []
        last_features_space = input_space
        for layer_size in hidden_layers:
            layers += [nn.Linear(last_features_space, layer_size), activation]
            last_features_space = layer_size
        layers += [nn.Linear(last_features_space, 1)]
        self.head = nn.Sequential(*layers)

    def get_value(self, features):
        return self.head(features)


class ActorCriticModel(nn.Module):
    def __init__(self, feature_extractor, action_dim, hidden_layers, discrete=True, activation=nn.ReLU()):
        super(ActorCriticModel, self).__init__()
        # action mean range -1 to 1
        self.features = feature_extractor
        if discrete:
            self.actor = DiscreteActor(self.features.features_space, action_dim, hidden_layers, activation)
        else:
            self.actor = CountinousActor(self.features.features_space, action_dim, hidden_layers, activation)
        self.critic = Critic(self.features.features_space, hidden_layers, activation)

    def get_action_dist(self, x):
        features = self.features(x)
        return self.actor(features)

    def forward(self, x):
        features = self.features(x)
        dist = self.actor.get_dist(features)
        value = self.critic.get_value(features)
        return dist, value