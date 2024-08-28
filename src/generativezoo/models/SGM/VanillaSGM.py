###################################################################################################################
####### Code based on https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing #######
###################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from tqdm import trange, tqdm
from config import models_dir
import os
from scipy import integrate
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score
import wandb

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'VanillaSGM')):
    os.makedirs(os.path.join(models_dir, 'VanillaSGM'))

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

def padding_factor(input_size):
  '''
  Padding factor correction required for the transpose convolution layers
  if h3 output is not even, we cannot add padding to tconv4
  '''
  h3_size = (((input_size - 2) // 2 - 1)//2 - 1)
  if h3_size % 2 == 0:
    return 1
  else:
    return 0

class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, in_channels=1, input_size=28):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(in_channels, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, output_padding=padding_factor(input_size))
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], in_channels, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))   
    # Encoding path
    h1 = self.conv1(x)  
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

def marginal_prob_std(t, sigma, device):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    The standard deviation.
  """    
  #t = torch.tensor(t, device=device)
  t = t.clone().detach().to(device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor((sigma**t).clone().detach(), device=device)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss

def outlier_score(model, x, marginal_prob_std, eps=1e-5):
  t_list = [0.001, 0.01, 0.05, 0.1, 0.2]
  for t in t_list:
    random_t = torch.ones(x.shape[0], device=x.device) * t
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3))
    if t == 0.001:
      loss_sum = loss
    else:
      loss_sum += loss
  
  return loss_sum/len(t_list)

def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=500, 
                           device='cuda', 
                           eps=1e-3,
                           channels=1,
                           input_size=28):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, channels, input_size, input_size, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):     
      batch_time_step = torch.ones(batch_size, device=device) * time_step 
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x

def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3,
               channels=1,
               input_size=28):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, channels, input_size, input_size, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps, desc = 'PC sampling', leave=False):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean

def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                atol=1e-6, 
                rtol=1e-6, 
                device='cuda', 
                z=None,
                eps=1e-3,
                channels=1,
                input_size=28):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, channels, input_size, input_size, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x

class VanillaSGM(nn.Module):
  def __init__(self, args, in_channels=1, input_size=28):
    '''
    Vanilla SGM model
    Args:
      args: The arguments for the model
      in_channels: The number of input channels.
      input_size: The size of the input image.
    '''
    super(VanillaSGM, self).__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.sigma = args.sigma
    self.n_epochs = args.n_epochs
    self.lr = args.lr
    self.model_channels = args.model_channels
    self.embed_dim = args.embed_dim
    self.in_channels = in_channels
    self.input_size = input_size
    self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma, device=self.device)
    self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=args.sigma, device=self.device)
    self.model = ScoreNet(self.marginal_prob_std_fn, self.model_channels, self.embed_dim, in_channels=in_channels, input_size=input_size).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
    self.dataset = args.dataset
    self.sample_and_save_freq = args.sample_and_save_freq
    self.num_steps = args.num_steps
    self.snr = args.snr
    self.sampler_type = args.sampler_type
    self.atol = args.atol
    self.rtol = args.rtol
    self.eps = args.eps
    self.no_wandb = args.no_wandb

  def train_model(self, dataloader, verbose=True):
    '''
    Train the Vanilla SGM model
    Args:
      dataloader: A PyTorch dataloader that provides training data.
    '''
    epoch_bar = trange(self.n_epochs, desc='Average loss: N/A')
    best_loss = np.inf

    create_checkpoint_dir()

    for epoch in epoch_bar:
      avg_loss = 0.0
      
      for x,_ in tqdm(dataloader, desc='Batches', leave=False, disable=not verbose):
        x = x.to(self.device)

        self.optimizer.zero_grad()
        loss = loss_fn(self.model, x, self.marginal_prob_std_fn)

        loss.backward()
        self.optimizer.step()

        avg_loss += loss.item()*x.shape[0]

      epoch_bar.set_description('Average Loss: {:5f}'.format(avg_loss / len(dataloader.dataset)))
      if not self.no_wandb:
        wandb.log({'loss': avg_loss / len(dataloader.dataset)}, step = epoch)

      if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(self.model.state_dict(), os.path.join(models_dir,'VanillaSGM', 'VanSGM_' + self.dataset + '.pt'))

      if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:

        samples = pc_sampler( self.model,
                              self.marginal_prob_std_fn,
                              self.diffusion_coeff_fn,
                              batch_size=16,
                              num_steps=self.num_steps,
                              snr=self.snr,
                              device=self.device,
                              channels=self.in_channels,
                              input_size=self.input_size)
        
        samples = samples*0.5 + 0.5
        samples = samples.clamp(0.0, 1.0)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(16)))

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(sample_grid.permute(1, 2, 0).cpu().numpy(), vmin=0.0, vmax=1.0)
        plt.axis('off')
        if not self.no_wandb:
          wandb.log({'samples': fig}, step = epoch)
        plt.close(fig)
  
  @torch.no_grad()
  def sample(self, num_samples=64):
    '''
    Sample from the Vanilla SGM model
    Args:
      num_samples: The number of samples to generate.
      num_steps: The number of sampling steps.
      snr: The signal-to-noise ratio for the PC sampler.
      sampler_type: The type of sampler. One of 'EM', 'PC', and 'ODE'.
    '''
    self.model.eval()
    if self.sampler_type == 'EM':
      samples = Euler_Maruyama_sampler(self.model, 
                                        self.marginal_prob_std_fn,
                                        self.diffusion_coeff_fn,
                                        batch_size=num_samples, 
                                        num_steps=self.num_steps, 
                                        device=self.device,
                                        eps=self.eps,
                                        channels=self.in_channels,
                                        input_size=self.input_size)
    elif self.sampler_type == 'PC':
      samples = pc_sampler(self.model, 
                            self.marginal_prob_std_fn,
                            self.diffusion_coeff_fn,
                            batch_size=num_samples, 
                            num_steps=self.num_steps, 
                            snr=self.snr, 
                            device=self.device,
                            eps=1e-3,
                            channels=self.in_channels,
                            input_size=self.input_size)
      
    elif self.sampler_type == 'ODE':
      samples = ode_sampler(self.model,
                            self.marginal_prob_std_fn,
                            self.diffusion_coeff_fn,
                            batch_size=num_samples,
                            atol=self.atol,
                            rtol=self.rtol,
                            device=self.device,
                            z=None,
                            eps=self.eps,
                            channels=self.in_channels,
                            input_size=self.input_size)

    samples = samples*0.5 + 0.5  
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(num_samples)))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(sample_grid.permute(1, 2, 0).cpu().numpy(), vmin=0.0, vmax=1.0)
    plt.axis('off')
    plt.show()

  @torch.no_grad()
  def outlier_detection(self, val_loader, out_loader):
    '''
    Detect outliers with a score-based model.
    Args:
      val_loader: A PyTorch dataloader that provides validation data.
      out_loader: A PyTorch dataloader that provides outlier data.
    '''
    self.model.eval()
    val_scores = []
    for x,_ in val_loader:
      x = x.to(self.device)
      score = outlier_score(self.model, x, self.marginal_prob_std_fn)
      val_scores.append(score.cpu().numpy())
    val_scores = np.concatenate(val_scores, axis=0)
    out_scores = []
    for x,_ in out_loader:
      x = x.to(self.device)
      score = outlier_score(self.model, x, self.marginal_prob_std_fn)  
      out_scores.append(score.cpu().numpy())
    out_scores = np.concatenate(out_scores, axis=0)
    y_true = np.concatenate([np.zeros_like(val_scores), np.ones_like(out_scores)], axis=0)
    y_score = np.concatenate([val_scores, out_scores], axis=0)
    auc_score = roc_auc_score(y_true, y_score)
    if auc_score < 0.2:
      auc_score = 1. - auc_score
    print('AUC score: {:.4f}'.format(auc_score))
    plt.figure(figsize=(10, 5))
    plt.hist(val_scores, bins=50, alpha=0.5, label='In-distribution')
    plt.hist(out_scores, bins=50, alpha=0.5, label='Out-of-distribution')
    plt.xlabel('Mean Score')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()