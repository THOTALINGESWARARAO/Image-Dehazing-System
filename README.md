Image Dehazing System using Residual U-Net

A deep learning project for restoring visibility in hazy and foggy images. The project explores a Residual U-Net built with PyTorch to predict an image correction and add it to the hazy input, with the goal of recovering clearer scenes while preserving useful image details.



Repository: THOTALINGESWARARAO/Image-Dehazing-System
Notebook: UNet_Image_Dehazing.ipynb

Overview

Haze and fog reduce image contrast, obscure textures, and make objects harder to distinguish. This project investigates a convolutional neural network that learns to transform hazy images into clearer outputs using paired synthetic examples and experiments with real-world fog images.

The main implementation is provided in a Google Colab notebook. It includes dataset preparation, a PyTorch model, loss-function experiments, and a training loop.

Key Features

Residual U-Net architecture: encoder-decoder network with skip connections and residual image correction.

PyTorch implementation: model definition, optimization, and training are implemented with PyTorch.

Synthetic hazy/clear pairs: uses the RESIDE-6K dataset structure, with hazy input images and GT clear ground-truth images.

Real-world haze experiments: includes preparation steps for O-HAZE and real fog imagery.

Combined loss experiments: combines pixel-level L1 loss, an SSIM-based structural loss, and VGG-based perceptual loss for paired synthetic data.

Google Colab support: the notebook checks CUDA availability and can use a GPU when one is enabled in the runtime.

Model Architecture

The notebook defines a ResidualUNet with three main stages:

Encoder: convolution blocks extract visual features at progressively lower spatial resolutions.

Decoder: transposed convolutions upsample features, while skip connections combine encoder and decoder features to retain spatial details.

Residual output: a final convolution predicts a correction that is added to the original input: output = input + residual.

The network operates on three-channel RGB images. The notebook's implementation uses convolution blocks with ReLU activations, max-pooling for downsampling, and transposed convolutions for upsampling.

Technology Stack

Python

PyTorch

Torchvision

Pillow (PIL)

pytorch-msssim for SSIM

Google Colab and Google Drive for notebook execution and dataset storage

Kaggle API for dataset download

Datasets

RESIDE-6K

The notebook downloads the kmljts/reside-6k dataset from Kaggle and extracts a training directory containing paired images:

RESIDE-6K/
└── train/
    ├── hazy/   # Hazy input images
    └── GT/     # Corresponding clear ground-truth images

The dataset is used for supervised dehazing experiments. Download access and dataset organization can vary, so check the dataset page and adjust the paths in the notebook if necessary.

Dataset page: RESIDE-6K on Kaggle

O-HAZE and real fog images

The notebook also contains steps to download and inspect O-HAZE images and to prepare a folder for real fog images. The O-HAZE archive shown in the notebook contains hazy, clear, and hayzed directories. Verify and configure the actual paths before combining these images with the training dataset; folder setup alone does not mean these images have been included in a successful training run.

Dataset page used in the notebook: O-HAZE on Kaggle

Please review each dataset's license and terms before redistribution. Do not upload the full datasets to this repository unless the dataset license permits it.

Training Approach

The notebook experiments with a combined loss for paired synthetic images:

L1 loss: encourages pixel-level similarity to the clear target.

SSIM loss: 1 - SSIM, encouraging structural similarity.

Perceptual loss: compares feature representations extracted from a pretrained VGG16 network.

The combined loss is defined in the notebook as a weighted sum:

combined_loss = 0.7 * L1_loss + 0.2 * (1 - SSIM) + 0.1 * perceptual_loss

The training configuration in the notebook uses the Adam optimizer with a learning rate of 1e-4, a StepLR scheduler with a step size of 10 and gamma of 0.5, and a configured run of 20 epochs. These are the notebook's intended settings; actual completed training depends on running the cells successfully in the correct order.

Run the Project

Option 1: Run in Google Colab

Open the notebook using the Open in Colab badge above.

Select Runtime → Change runtime type and choose a GPU runtime if one is available.

Run the cells in order, from dataset setup through model definition and training.

When prompted, mount Google Drive and configure the dataset paths to match your Drive folders.

Configure Kaggle credentials securely if using the Kaggle API. Do not commit kaggle.json or API tokens to GitHub.

Review the training logs and save the trained checkpoint and output images to an appropriate folder.

Option 2: Run locally

A local Python environment may be used, but the notebook currently contains Google Colab and Google Drive-specific paths. To run it locally, first convert those paths to local directories and remove or adapt Colab-only commands.

Install the core dependencies:

pip install torch torchvision pillow pytorch-msssim

Exact PyTorch installation commands depend on your operating system and whether you want CPU or CUDA support. Install a compatible PyTorch build from the official PyTorch installation selector: https://pytorch.org/get-started/locally/.

Repository Structure

Image-Dehazing-System/
├── README.md
└── UNet_Image_Dehazing.ipynb

The notebook creates additional data, model, utility, and output folders in Google Drive during execution. Those runtime folders are not necessarily present in this GitHub repository.

Evaluation and Results

The notebook includes experiments with image-quality losses, but it does not currently establish a reproducible final benchmark in the committed notebook output. Therefore, final PSNR, SSIM, or MSE values should be added here only after the training and evaluation cells run successfully on a clearly specified test set.

For a meaningful evaluation, report:

PSNR (Peak Signal-to-Noise Ratio): pixel-level reconstruction quality.

SSIM (Structural Similarity Index): similarity in luminance, contrast, and structure.

MSE (Mean Squared Error): average squared difference between the restored image and the ground truth.

When publishing results, document the test dataset, image preprocessing, checkpoint used, and evaluation code so the numbers can be reproduced.

Current Status and Notes

This repository contains an experimental notebook rather than a packaged application. In the saved notebook output, some cells show setup or training errors, including a training run where fog_loss is not defined at execution time. Before treating the model as trained, run the notebook from a clean runtime, execute the definitions before they are used, verify the dataset paths and paired-image alignment, and confirm that the final training loop completes and saves a checkpoint.

Possible Future Improvements

Complete and validate the training and evaluation pipeline.

Add reproducible train/validation/test splits and checkpoint saving.

Save before-and-after visual comparisons for representative test images.

Add inference code that accepts an arbitrary hazy image and writes the restored result.

Track PSNR, SSIM, and MSE consistently on a held-out test set.

Package configuration and dependencies in a requirements.txt file.

Consider a simple web interface or API for image upload and dehazing.

Author

Thota Lingeswara Rao
GitHub: @THOTALINGESWARARAO

Acknowledgements

RESIDE-6K dataset contributors.

O-HAZE dataset contributors.

The PyTorch and Google Colab communities.
