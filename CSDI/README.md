# Implementation of CSDI model

Here it is my implementation of the **CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation** paper.
Link: https://arxiv.org/pdf/2107.03502.pdf

# Implementation details
The model has been implemented according the details provided in the paper.  The model can be initialized as the following

    model = CSDI(noise_steps=50,
			     l = L,
			     fs = FS,
			     beta_start=0.0001,
			     beta_end=0.5,
			     temp_strips_blocks = 1,
			     feat_strips_lenght = 1,
			     num_features = 4,
			     num_res_blocks = 4,
			     number_heads = 8,
			     model_dim = 64,
			     emb_dim = 128,
			     time_dim = 128,
			     feat_dim = 16,
			     do_prob = 0.1).to(device)

Initializing the model as above, (with the hyperparameters used in the paper, it can be seen how the number of trainable parameters matches with the one specified in the paper.

<p align="center">
  <img src="https://user-images.githubusercontent.com/72130704/208125119-9093cff2-76f9-421e-808a-85527514e293.png">
</p>


The following options have been implemented in order to to alleviate computational requirements:

 - ***temp_strips_blocks***: For spliting the signals in different strips before feeding them into the temporal transformer block.
 - ***feat_strips_lenght***:  For splitting the signal in strips of the lenght specified by this parameter before feeding them into the feature transformer block.

In addition to this, the following parameters have to be given to the model:

 - ***L***: Lenght of the signal (in seconds)
 - ***FS***: Frequency Sample of the signal.
