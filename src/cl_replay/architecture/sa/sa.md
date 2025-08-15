* MNIST VAE: two hidden layers 256-512, latent space z has dimensions 8
  * Bernoulli distribution over the pixels as the decoder output distribution
  * standard Gaussian as the prior
  * class-conditioning by appending one-hot encoding vector to the encoder and decoder inputs
  * VAE is trained for 100K steps, and forgetting training is trained for 10k steps
  * learning rate of 10^-4, batch_size = 256
  * Sample 50K samples to calculate the FIM
  * sample the replay data from a frozen copy of the original VAE during forgetting training
  
* Classifier: two-layer CNN for 20 epochs





Forget training:
10k steps, 10^-4 LR, batch-size=256, use 50k samples to calculate the FIM
