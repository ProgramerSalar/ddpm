import torch 
from ddpm import DDPMSampler
import numpy as np 
import tqdm

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sets the dimensions of the generated image 
#     |
# create a generate function (prompt, uncond_prompt=None,  input_image=None, strength=0.8,  do_cfg = True, 
# cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, seed=None, device=None, idle_device=None, tokenizer=None,)
#     |
# Disable Gradient calculation [A]
#     |
# Validate Strength: Checks if strength is between 0 and 1. If not, raises a ValueError.
#     |
# Device Handling: Defines a lambda function to_idle to move models to the idle_device if specified, 
# or leave them on the current device.
#     |
# Initializes a random number generator on the specified device 
#     |
# if seed is none return the generator seed otherwise generate the manual seed with seed
#     |
# Load CLIP Model and Moves the CLIP model to the specified device.
#     |
# 1. [check if classifier-free-Gudance (cfg) is Enabled]
#     |
# 2. Tokenize the conditional tokens pass the batch_encode_plus function in tokenizer
# and give the prompt, padding is max_length and  max_length is 77
#     |
# 3. Convert Tokens to Tensor
#     |
# 4. Get Conditional Context to clip function pass the conditional tokens 
#     |
# 5. Tokenize the Unconditional Prompt
#     |
# 6. Convert Unconditional Tokens to Tensor
#     |
# 7. Get Unconditional Context to clip function pass the unconditional tokens 
#     |
# 8. Concatenate Contexts

# ***********************************************************************************************************************************


# 1. Else Block Start:
#     |
# 2. Tokenize the Prompt
#     |
# 3. Convert Tokens to Tensor
#     |
# 4. Get Context to clip function pass the tokens 
#     |
# 5. Move clip to idle_device





# [condition] if else 
#     |
# 1. Check Sampler Name is equal to ddpm
#     |
# 2. Initialize DDPM Sampler,  fn DDPMSampler pass the generator 
#     |
# 3. Set Inference Timesteps: object sampler pass the function set_inference_timesteps in n_inference_steps
#     |
# 4. Handle Unknown Sampler Name: raise a value error ("Unknow sampler value %s. ")




# ************************************************************************************************************************************************************************************************************


# 1. Define Latent Shape: (Batch_size, 4, height / 8, width / 8)
#     |
# 2. Check if Input Image is Provided:
#     |
# 3. Load and Prepare Encoder Model:
#     |
# 4. Resize Input Image: object is input_image_tensor pass the size=(HEIGHT, WIDTH)
#     |
# 5. Convert Image to NumPy Array:
#     |
# 6. Convert Image to PyTorch Tensor: dtype=float32
#     |
# 7. Rescale Image Values: old_range=(0, 255) new_range=(-1, 1)
#     |
# Unsqueeze(0): Adds a batch dimension to the tensor, changing its shape to (Batch_size, Height, Width, Channels).
#     |
# Permute: Rearranges the dimensions of the tensor to match the shape (Batch_size, Channels, Height, Width).
#     |
# Random Noise: Generates a tensor of random noise and object is encoded_noise and take a torch.randn size=latent_shape, generator=generator 
#     |
# Encoding: object is latents =  Pass the input_image_tensor and encoded_noise in the encoder 
#     |
# Set Strength: object sampler function set_strength pass the strength
#     |
# Add Noise: latents =  object sampler function add_noise pass latents, sampler.timesteps[0]
#     |
# Device Handling: move encoder to to_idle 
#     |
# [Else]
#     |
# Random Noise: latents = torch.randn pass the latent_shape, generator

# ************************************************************************************************************************************************************************************************************************************************************************
#  diffusion = models["diffusion"]
#  diffusion.to(device)
#     |
# Initialize Timesteps: tqdm(sampler.timesteps)
#     |
# Iterate: Loops through each timestep in the timesteps list. i is the index of the current timestep.
# for i, timestep in enumerate(timesteps)
#     |
# Generate Time Embedding: time_embedding = function get_time_embedding pass the timestep to device 
#     |
# Set Input: Assigns the latents tensor as the initial model input
#     |
# if do_cfg
#     |
# model_input: obj model_input function repeat(2, 1, 1, 1) 
#     |
# Predict Noise:model_output= Passes the model_input, context, and time_embedding through the diffusion model to predict the noise. The output is a tensor of the same shape as the input.

# *******************************************************************************************

# if do_cfg
#     |
# object output_cond, output_uncond = model_output.chunk(2)
#     |
# object model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
#     |
# object latents = sampler.step(timestep, latents, model_output)
#     |
# to_idle(diffusion)

# ********************************************************************************************************

# Load Decoder: Loads the decoder model from the models dictionary.
#     |
# Move to Device: Moves the decoder model to the specified device (e.g., CPU or GPU).
#     |
# Decode Latents object images fn decoder pass the latents
#     |
# Move Decoder to Idle Device:
#     |
# Rescale Image Values: old_range=(-1, 1) new_range=(0, 255), clamp=True
#     |
# Permute Tensor Dimensions:(0, 2, 3, 1)
#     |
# Convert Tensor to NumPy Array:images = images.to("cpu", torch.uint8).numpy()
#     |
# Return First Image: return images[0]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




# sets the dimension of the generated image 
WIDTH = 512 
HEIGHT = 512 
LATENT_HEIGHT = HEIGHT // 8 
LATENT_WIDTH = HEIGHT // 8 



def generate(prompt,
             uncond_prompt=None,
             input_image=None,
             strength=0.8,
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):
    


    # disable gradient calculation
    with torch.no_grad():

        # validate strength: check if strength is between 0 and 1 
        if 0 < strength <= 1:
            raise ValueError("strength is between 0 and 1")
        

        if idle_device:
            to_idle = lambda x: x.to(device)

        else:
            to_idle = lambda x: x 


        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()


        else:
            generator.manual_seed(seed)


        clip = models["clip"]
        clip.to(device)

        if do_cfg:

            # tokenize the conditional tokens
            cond_token = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            )

            # convert tokens to tensor 
            cond_token = torch.tensor(data=cond_token, dtype=torch.tensor, device=device)

            # get conditional context 
            cond_token = clip(cond_token)

            # tokenize the uncond prompt 
            uncond_token = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            )

            # convert unconditional tokens to tensor 
            uncond_token = torch.tensor(data=uncond_token, dtype=torch.tensor, device=device)

            # get unconditional context to clip function pass the unconditional tokens 
            uncond_token = clip(uncond_token)


            # concatenate contexts 
            context = torch.cat([cond_token, uncond_token])


        else:

            context = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            )

            # convert tokens to tensor 
            context = torch.tensor(data=context, dtype=torch.tensor, device=device)

            context = clip(context)

            idle_device(clip)


        
        if sampler_name == "ddpm":

            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)

        else:

            raise ValueError("Unknown sampler value %s")
        


        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:

            encoder = models["encoder"]
            
            # Resize the input images 
            input_image_tensor = input_image.resize(size=(HEIGHT, WIDTH))

            # convert image to numpy array 
            input_image_tensor = np.array(input_image_tensor)

            # convert pytorch tensor 
            input_image_tensor = torch.tensor(data=input_image_tensor, dtype=torch.float32)

            # Rescale image 
            input_image_tensor = rescale(input_image_tensor, old_range=(0, 255), new_range=(-1, 1))

            # unsqueeze
            input_image_tensor = input_image_tensor.unsqueeze(0)

            # permute 
            input_image_tensor = torch.permute(input_image_tensor, dims=(0, 3, 1, 2))

            # Random noise 
            encoded_noise = torch.randn(size=latent_shape, 
                                        generator=generator,
                                        device=device)
            


            # Encoding object 
            latents = encoder(input_image_tensor, encoded_noise)

            # set strength 
            sampler.set_strength(strength)

            # add noise 
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # device Handling 
            to_idle(encoder)



        else:

            # random noise 
            latents = torch.randn(latent_shape, generator=generator, device=device)



        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for i, timestep in enumerate(timesteps):

            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:

                model_input = model_input.repeat(2, 1, 1, 1)

            # predict the noise 
            model_output = diffusion(model_input, context, time_embedding)



            if do_cfg:

                output_cond, output_uncond = model_output.chunk(2)

                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)


        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        idle_device(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        images = torch.permute(images, dims=(0, 2, 3, 1))

        # convert tesnro to numpy array 
        images = images.to("cpu", torch.uint8).numpy()



        return images[0]






      

    

# --------------------------------------------------------------------------------------------------------
#         ## Rescale ##
# parameter (x, old_range, new_range, clamp=False)
#     |
# Extract Old and New range values 
#     |
# Normalize `x` to a 0-1 range relative to `old_range`
#     |
# scale `x` from 0-1 to the 0-1 Range of `new_range`
#     |
# Adjust `x` to the New Range 
#     |
# Optionally Clamp `x` within the New Range 
#     |
# return rescaled value 
# ----------------------------------------------------------------------------------------------------------



def rescale(x, old_range, new_range, clamp=False):

    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp: 
        x = x.clamp(new_min, new_max)

    return x 

# --------------------------------------------------------------------
#     ## Get time embedding ## 
# calculate frequencies 
#     |
# compute scaled Time step 
#     |
# generate embeddings 
# -------------------------------------------------------------------------


def get_time_embedding(timesteps):

    # shape: (160,)  # Position
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)

    # shape: (1, 160)
    x = torch.tensor([timesteps], dtype=torch.float32)[:, None] * freqs[None]

    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



