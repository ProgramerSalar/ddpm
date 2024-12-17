

import torch 
import numpy as np 


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# class DDPMSampler 
# init ( generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.120)
#     |
# define betas using the linespace  dtype=float32
#     |
# define alphas (1 - betas)
#     |
# object alpha cumpulative product input=alphas dim=0
#     |
# one tensor 
#     |
# define generator and num_train_timesteps
#     |
# object timesteps (range array of numpy to reverse and copy) and convert to numpy 
# ______________________________________________________________________________________________________________________________________

# function set_inference_timesteps (num_inference_steps=50)
#     |
# step_ratio (1000 / 50) = 20 
#     |
# object timesteps = range of numpy (0 -> 50) * 20.round() reverse copy astype(np.int64)
#     |
# self.timesteps = convert timesteps to numpy 
# ________________________________________________________________________________________________________________________________________



# function _get_previous_timesteps (timestep: int)
#     |
# object prev_t = timestep - 1000 // 50 
#     |
# return prev_t


# __________________________________________________________________________________________________________________________________________

# function _get_variance(timestep:int)
    # |
# object prev_timestep = _get_previous_timesteps(timestep)
    # |
# alpha_prod_t = self.alphas_cumprod[timestep]
#     |
# calculate alpha product timestep previous = alpha cumulative product in previous timesteps if prev_timestep >=0 else self.one 
    # |
# calculate current beta timestep = 1 - alpha_product timestep / alpha_product timestep previous 
    # |
# calculate variance = (formula is 7)
    # |
# clamp the value min=1e-20
    # |
# return variance

# ___________________________________________________________________________________________________________________________________________

# function step (timestep: int, latents: torch.Tensor, model_output: torch.Tensor)
#    |
# t = timestep
#     |
# previous timestep = self._get_previous_timesteps(t)
#     |
# ## 1. compute alphas, betas 
# alpha_prod_t = alpha_cumprod at timesteps
#     |
# alpha_product timestep previous = alpha cumulative product at previous time if previous timestep >= 0 else one 
#     |
# beta product timestep = 1 - alpha product timestep
#     |
# beta product timestep previous = 1 - alpha product timestep previous 
#     |
# current alpha timestep = alpha product timestep / alpha product timesteps previous
#     |
# current beta timestep = 1 - current alpha timestep
#     |
# ## 2. compute original sample as formula 15 
#     |
# compute  formula - 7 
# predicted original sample coefficient = root of alpha product timesteps previous * current beta timesteps / beta product timestep 
#     |
# current sample coefficient = root of current alpha timestep * beta product timestep previous / beta product timesteps 
#     |
#     |
# # cumpute the mean 
# pred_prev_sample = predicted original sample coefficient * predicted original sample + current sample coefficient * latents 
#     |
#     v
# ## add noise 
# variance = 0 
# if t > 0: 
#     |
# device = model_output.device 
#     |
# noise = random number of torch (shape of model output, device, dtype=model output.dtype)
#     |
# compute the variance root of _get_variance at timestep * noise 
#     |
# predicted previous sample = predicted previous sample + variance 
#     |
# return pred_prev_sample

# ______________________________________________________________________________________________________________________________________________________________________________________


# function add noise (original samples = floatTensor, timesteps = Inttensor) -> FloatTensor
#     |
# alpha cumulative product to (device = original sample.device, dtype)
#     |
# timesteps = timesteps to (original sample device)
#     |
# sqrt alpha product = root of alpha cumulative product at timesteps 
#     |
# flatten 
#     |
# while length shape of sqrt alpha product < length shape of original samples 
#     |
# sqrt alpha product = sqrt alpha product unsqueeze(-1)
#     |
# sqrt one minus alpha product = root of 1 - alpha cumproduct at timesteps 
#     |
# flatten 
#     |
# while length shape of sqrt one minus alpha product < length shape of original samples 
#     |
# sqrt one minus alpha product = sqrt one minus alpha product unsqueeze(-1)
#     |
# #calculate equation - 4 
# noise = torch.randn(original sample shape, generator, original sample device, same dtype)
#     |
# noisy_samples = sqrt alpha product * original_samples + sqrt alpha product * noise 
#     |
# return noisy_samples

# _______________________________________________________________________________________________________________________________________________________________________________________

# function set_strength(strength=1)
#     |
# start_step = 50 - int(50 * strength)
#     |
# timesteps = timesteps[start_step:]
#     |
# self.start_step = start_step
# __________________________________________________________________________________________________________________________________


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class DDPMSampler:

    def __init__(self,
                 generator,
                 num_training_steps=1000,
                 beta_start=0.00085,
                 beta_end=0.120):
        


        self.betas = torch.linspace(start=beta_start ** 0.5, 
                                    end=beta_end ** 0.5, 
                                    steps=num_training_steps, 
                                    dtype=torch.float32)
        


        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(input=self.alphas,
                                           dim=0)
        
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_train_timesteps = num_training_steps

        self.timesteps = torch.from_numpy(np.arange(start=0, stop=num_training_steps)[::-1].copy())




    def set_inference_timesteps(self, num_inference_steps=50):

        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // self.num_inference_steps 
        timesteps = (np.arange(start=0, stop=self.num_inference_steps) * 20).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)



    def _get_previous_timesteps(self, timestep: int):

        prev_t = (timestep - self.num_train_timesteps) // self.num_inference_steps
        return prev_t
    

    


    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):

        self.t = timestep

        self.prev_t = self._get_previous_timesteps(self.t)

        # 1. compute alphas, betas 
        alpha_prod_t = self.alpha_cumprod[self.t]
        alpha_prod_t_prev = self.alpha_cumprod[self.prev_t] if self.prev_t >= 0 else self.one
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t


        # compute original sample as formula 15 
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / torch.sqrt(alpha_prod_t)

        # compute the formula (7)
        pred_original_sample_coeff = (torch.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        current_sample_coeff = (torch.sqrt(current_alpha_t) * beta_prod_t_prev) / beta_prod_t

        # compute mean 
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents


        # add noise 
        variance = 0 
        if self.t > 0:

            device = model_output.device
            noise = torch.randn(size=model_output.shape,
                                device=model_output.device,
                                dtype=model_output.dtype)
            

            variance = self._get_variance(timestep).sqrt() * noise
        
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample


    def _get_variance(self, timestep:int):

        prev_t = self._get_previous_timesteps(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]

        alpha_prod_t_prev = self.alpha_cumprod[self.prev_t] if prev_t >= 0 else self.one 
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(input=variance, 
                               min=1e-20)
        
        return variance




    
    def add_noise(self,
                  original_samples=torch.FloatTensor,
                  timesteps=torch.IntTensor) -> torch.FloatTensor:
        

        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, 
                                              dtype=original_samples.dtype)
        
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_prod = torch.sqrt(input=alpha_cumprod[timesteps])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alpha_cumprod[timesteps]).flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # calculate equation - 4 
        noise = torch.randn(original_samples.shape,
                            generator=self.generator,
                            device=original_samples.device, 
                            dtype = original_samples.dtype)
        


        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_alpha_prod * noise 

        return noisy_samples
    

    

    def set_strength(self, strength=1):

        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        timesteps = timesteps[start_step:]
        self.start_step = start_step


        






        







        











        



    

        