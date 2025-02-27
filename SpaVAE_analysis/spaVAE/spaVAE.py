import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy as np
import pandas as pd
from .SVGP import SVGP
from .I_PID import PIDControl
from .VAE_utils import *
from collections import deque

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss
        
class SPAVAE(nn.Module):
    def __init__(self, input_dim, GP_dim, Normal_dim, encoder_layers, decoder_layers, noise, encoder_dropout, decoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    KL_loss, dynamicVAE, init_beta, min_beta, max_beta, dtype, device):
        super(SPAVAE, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.input_dim = input_dim
        self.PID = PIDControl(Kp=50, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss          
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta           
        self.dtype = dtype
        self.GP_dim = GP_dim            
        self.Normal_dim = Normal_dim    
        self.noise = noise              
        self.device = device
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([GP_dim+Normal_dim]+decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim))
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, input_dim))

        self.MSE_loss = lambda x,y: 100 * torch.sum(((x - y)) ** 2)
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x, y, num_samples=1):
        def kl_divergence_gaussian_laplace(gaussian_mu, gaussian_var, laplace_scale=1.0):
            gaussian_sigma = torch.sqrt(gaussian_var)
            kl_term = (
                torch.log(torch.tensor(2.0 * laplace_scale, device=gaussian_mu.device)) + 
                torch.abs(gaussian_mu) / laplace_scale + 
                (gaussian_var / (2 * laplace_scale**2)) - 
                0.5 - torch.log(gaussian_sigma)
            )
            return kl_term.sum()
        
        self.train()
        b = y.shape[0]
        qnet_mu, qnet_var = self.encoder(y)

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # cross entropy term
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var)
        gp_ce_term = torch.sum(gp_ce_term)

        # KL term of GP prior
        gp_KL_term = gp_ce_term - inside_elbo


        gaussian_KL_term = kl_divergence_gaussian_laplace(gaussian_mu, gaussian_var).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        recon_loss = 0
        for f in latent_samples:
            hidden_samples = self.decoder(f)
            mean_samples_ = self.dec_mean(hidden_samples)

            mean_samples.append(mean_samples_)
            recon_loss += self.MSE_loss(mean_samples_, y)
        recon_loss = recon_loss / num_samples
        
        elbo = recon_loss + self.beta * (gp_KL_term + gaussian_KL_term)
        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, 0, 0, [], [], qnet_mu, qnet_var, \
            mean_samples, disp_samples, [], [], latent_samples, 0


    def batching_latent_samples(self, X, Y, batch_size=512):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        """ 

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)

        latent_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(ybatch)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
#            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

#             # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        return latent_samples.numpy()


    def batching_recon_samples(self, Z, batch_size=512):
        self.eval()

        Z = torch.tensor(Z, dtype=self.dtype)

        recon_samples = []

        num = Z.shape[0]
        num_batch = int(math.ceil(1.0*Z.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            zbatch = Z[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            h = self.decoder(zbatch)
            mean_batch = self.dec_mean(h)
            recon_samples.append(mean_batch.data.cpu().detach())

        recon_samples = torch.cat(recon_samples, dim=0)
        return recon_samples.numpy()

    def train_model(self, pos, abundances, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
            maxiter=5000, save_model=True, model_weights="model.pt", print_kernel_scale=True):
        self.train()

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(abundances, dtype=self.dtype))
        train_dataset = dataset

        if abundances.shape[0] > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        queue = deque()

        print("Training")

        recon_losses = []
        kld_losses = []
        epochs = []
        self.beta = 0
        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            laplacian_KL_term_val = 0
            num = 0
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                elbo, recon_loss, gp_KL_term, laplacian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg = \
                    self.forward(x=x_batch, y=y_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()
    
                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                laplacian_KL_term_val += laplacian_KL_term.item()

                num += x_batch.shape[0]

                if self.dynamicVAE:
                    KL_val = (laplacian_KL_term.item() + gp_KL_term.item()) / x_batch.shape[0]
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    self.beta, _ = self.PID.pid(self.KL_loss, avg_KL)
                    if len(queue) >= 10:
                        queue.popleft()


            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            laplacian_KL_term_val = laplacian_KL_term_val/num
            recon_losses.append(recon_loss_val)
            kld_losses.append((gp_KL_term_val + laplacian_KL_term_val))
            epochs.append(epoch)
            print('Training epoch {}, ELBO:{:.8f}, MSE Recon loss:{:.8f}, Laplacian KLD loss:{:.8f}, GP KLD Loss:{:.8f}'.format(epoch+1, elbo_val, recon_loss_val, laplacian_KL_term_val, gp_KL_term_val))
            print('Current beta', self.beta)
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if save_model:
                torch.save(self.state_dict(), model_weights)

    def train_model_multi(self, pos_datasets, abundances_datasets, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
            maxiter=5000, save_model=True, model_weights="model.pt", print_kernel_scale=True):
        """
        Model training.

        Parameters:
        -----------
        pos: array_like, shape (n_spots, 2)
            Location information.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        size_factor: array_like, shape (n_spots)
            The size factor of each spot, which need for the NB loss.
        lr: float, defalut = 0.001
            Learning rate for the opitimizer.
        weight_decay: float, default = 0.001
            Weight decay for the opitimizer.
        train_size: float, default = 0.95
            proportion of training size, the other samples are validations.
        maxiter: int, default = 5000
            Maximum number of iterations.
        patience: int, default = 200
            Patience for early stopping.
        model_weights: str
            File name to save the model weights.
        print_kernel_scale: bool
            Whether to print current kernel scale during training steps.
        """

        self.train()
        dataloaders = []
        for i in range(len(pos_datasets)):
            dataset = TensorDataset(torch.tensor(pos_datasets[i], dtype=self.dtype), torch.tensor(abundances_datasets[i], dtype=self.dtype))
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            dataloaders.append(dl)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        queue = deque()

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            laplacian_KL_term_val = 0
            # noise_reg_val = 0
            num = 0
            for dataloader in dataloaders:
                for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    elbo, recon_loss, gp_KL_term, laplacian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                        mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg = \
                        self.forward(x=x_batch, y=y_batch, num_samples=num_samples)

                    self.zero_grad()
                    elbo.backward(retain_graph=True)
                    optimizer.step()
        
                    elbo_val += elbo.item()
                    recon_loss_val += recon_loss.item()
                    gp_KL_term_val += gp_KL_term.item()
                    laplacian_KL_term_val += laplacian_KL_term.item()


                    num += x_batch.shape[0]

                    if self.dynamicVAE:
                        KL_val = (laplacian_KL_term.item() + gp_KL_term.item()) / x_batch.shape[0]
                        queue.append(KL_val)
                        avg_KL = np.mean(queue)
                        self.beta, _ = self.PID.pid(self.KL_loss, avg_KL)
                        if len(queue) >= 10:
                            queue.popleft()


            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            laplacian_KL_term_val = laplacian_KL_term_val/num
            # noise_reg_val = noise_reg_val/num

            print('Training epoch {}, ELBO:{:.8f}, MSE loss:{:.8f}, Laplacian KLD loss:{:.8f}, GP KLD Loss:{:.8f}'.format(epoch+1, elbo_val, recon_loss_val, laplacian_KL_term_val, gp_KL_term_val))
            print('Current beta', self.beta)
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if save_model:
                torch.save(self.state_dict(), model_weights)