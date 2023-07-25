# Style GAN 

This project implements various components of the Style Generative Adversarial Network (GAN) for image synthesis. The Style GAN is a popular generative model that can generate high-quality images and allows for control over the style and diversity of the generated samples.

## Components Implemented

The project includes the following components of the Style GAN:

1. **Truncation Trick**: The truncation trick allows controlling the trade-off between the quality and diversity of generated images by adjusting the truncation factor. It is used during the generation process to focus on higher-quality samples.

2. **Adaptive Instance Normalization (AdaIN)**: AdaIN is a normalization technique used to apply the style of one image to the content of another. It is employed in the Style GAN to combine the style and content information during the synthesis process.

3. **Mapping Network**: The mapping network is used to map a latent space vector to a more expressive space. It is used to control the style of generated images and is connected to the generator network.

4. **Progressive Growing**: Progressive growing is a training technique that starts with low-resolution images and gradually increases the resolution during training. This method helps stabilize the training process and allows the generator to learn details at different scales.

5. **Noise Injection**: Noise injection is the process of adding random noise to the generator's input or intermediate layers. It helps increase the diversity of generated images and reduces artifacts.


## Architecture

Folloiwng image combines all these components into one diagarm to make up the Style GAN architecture.

![Style GAN](style.png)

