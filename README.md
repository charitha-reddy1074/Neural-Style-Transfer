# Neural Style Transfer

A deep learning project that combines the **content of one image** with the **artistic style of another image** using Neural Style Transfer (NST).

This project uses a pre-trained **VGG19 convolutional neural network** to extract content and style representations from images. The generated image is optimized iteratively until it preserves the structure of the content image while reproducing the textures, colors, and visual patterns of the style image.

## Demo

The project includes an interactive interface built with **Gradio**, allowing users to upload:

- A content image
- A style image

The system then generates a stylized image by combining both inputs.

## How Neural Style Transfer Works

Neural Style Transfer treats the output image as an optimizable variable rather than training a new neural network from scratch.

The process consists of:

1. Loading a content image and a style image.
2. Passing both images through a frozen, pre-trained VGG19 model.
3. Extracting content features from a deep convolutional layer.
4. Extracting style features from multiple convolutional layers.
5. Computing content and style losses.
6. Updating the generated image using gradient descent.
7. Repeating the optimization process until the output image converges.

The final image maintains the semantic structure of the content image while adopting the artistic appearance of the style image.

## Technical Architecture

### Pre-trained VGG19 Network

The project uses VGG19 trained on the ImageNet dataset.

The VGG19 model is used only as a feature extractor:

```python
vgg.trainable = False
```

The network weights are not updated during optimization. Instead, the pixel values of the generated image are optimized.

### Content Representation

The content image is represented using a deep convolutional layer:

```text
block5_conv2
```

Deep layers capture high-level information such as:

- Object shapes
- Image composition
- Semantic structure
- Spatial arrangement
- Scene layout

The content loss compares the feature representation of the generated image with the feature representation of the content image.

### Style Representation

Style features are extracted from multiple layers:

```text
block1_conv1
block2_conv1
block3_conv1
block4_conv1
block5_conv1
```

These layers capture visual information at different scales:

- Low-level edges and colors
- Textures and brush patterns
- Shapes and local structures
- Higher-level artistic patterns

Using multiple layers allows the model to reproduce both fine-grained and coarse artistic details.

## Loss Functions

The optimization objective combines content loss and style loss.

### Content Loss

Content loss measures the difference between content features from the generated image and the original content image.

A common formulation is the mean squared error:

```text
J_content = MSE(F_generated, F_content)
```

A lower content loss indicates that the generated image is preserving the structure of the content image.

### Gram Matrix

Style is represented using Gram matrices.

For a feature map, the spatial dimensions are flattened and the correlations between feature channels are calculated:

```text
G = AᵀA
```

The Gram matrix captures:

- Texture
- Color relationships
- Brushstroke patterns
- Feature correlations
- Artistic appearance

Because spatial locations are aggregated, the Gram matrix represents style independently of the exact arrangement of objects in the image.

### Style Loss

Style loss compares the Gram matrices of the generated image and the style image across multiple VGG19 layers:

```text
J_style = Σ w_l × MSE(G_generated_l, G_style_l)
```

Each style layer contributes to the overall style loss using a layer-specific weight.

### Total Loss

The final optimization objective is a weighted combination:

```text
J_total = α × J_content + β × J_style
```

Where:

- `α` is the content weight
- `β` is the style weight
- A larger content weight preserves the original image structure
- A larger style weight produces stronger artistic effects

## Optimization

The generated image is represented as a TensorFlow variable:

```python
generated_image = tf.Variable(content_image)
```

TensorFlow's `GradientTape` calculates gradients of the total loss with respect to the image pixels:

```python
with tf.GradientTape() as tape:
    loss = total_loss(...)
    
gradients = tape.gradient(loss, generated_image)
optimizer.apply_gradients([(gradients, generated_image)])
```

The project uses the Adam optimizer to update the generated image over multiple iterations.

Important optimization parameters include:

- Learning rate
- Number of iterations
- Content loss weight
- Style loss weight
- Style-layer weights
- Image resolution
- Image initialization method

## Image Processing Pipeline

### Input Processing

Before images are passed to VGG19, they are processed through the following steps:

1. Load the image using PIL.
2. Resize it to a consistent resolution.
3. Convert it to a NumPy array.
4. Add a batch dimension.
5. Apply VGG19 preprocessing.

Typical tensor shape:

```text
(height, width, channels)
→
(1, height, width, channels)
```

### VGG19 Preprocessing

VGG19 expects images to follow the preprocessing convention used during ImageNet training.

The preprocessing includes:

- Converting RGB channel ordering to BGR
- Subtracting ImageNet channel means
- Preparing pixel values for the network

The ImageNet mean values are approximately:

```text
B: 103.939
G: 116.779
R: 123.680
```

### Image Deprocessing

After optimization, the processed image is converted back for display:

1. Add the ImageNet mean values.
2. Convert BGR back to RGB.
3. Clip values to the range `[0, 255]`.
4. Convert values to `uint8`.
5. Display or save the generated image.

## Datasets

This project does not require a dedicated paired style-transfer dataset.

Instead, it uses:

- A content image supplied by the user
- A style image supplied by the user
- VGG19 weights pre-trained on ImageNet

ImageNet is not used to train the style-transfer optimization directly. It provides the learned visual representations used by VGG19 for feature extraction.

## Evaluation Metrics

Neural Style Transfer does not have a single ground-truth output, so traditional classification metrics such as accuracy, precision, recall, and F1-score are not appropriate.

The following metrics can be used instead.

### Content Loss

Measures how well the generated image preserves the original content structure.

Lower values generally indicate better content preservation.

### Style Loss

Measures the difference between the generated image's style representation and the target style image.

Lower values generally indicate stronger style similarity.

### Total Loss

Tracks the combined optimization objective:

```text
Total Loss = Content Loss + Style Loss
```

A decreasing total loss usually indicates that the optimization process is progressing.

### Structural Similarity Index

SSIM can be used to compare the generated image with the content image.

It evaluates structural similarity based on:

- Luminance
- Contrast
- Local structure

A higher SSIM generally indicates stronger structural preservation, although excessive structural similarity may also indicate weak style transfer.

### Fréchet Inception Distance

FID can be used when evaluating a collection of generated images against a target artistic dataset.

Lower FID values indicate that the generated image distribution is closer to the reference artistic distribution.

FID is more useful for benchmarking many outputs than for evaluating a single image.

### Human Evaluation

Because artistic quality is subjective, human evaluation is also important.

Reviewers can evaluate:

- Content preservation
- Style fidelity
- Visual quality
- Color consistency
- Texture similarity
- Artifact suppression
- Overall aesthetic quality

Possible evaluation methods include Mean Opinion Score and blind A/B comparisons.

## Gradio Interface

The project integrates Gradio to provide an interactive interface.

Users can:

1. Upload a content image.
2. Upload a style image.
3. Run the style-transfer process.
4. View the generated image.

This makes the model accessible without requiring users to interact directly with the TensorFlow optimization code.

## Technologies Used

- Python
- TensorFlow
- Keras
- VGG19
- NumPy
- Pillow
- Matplotlib
- Gradio
- Jupyter Notebook
- ImageNet pre-trained weights

## Installation

Clone the repository:

```bash
git clone https://github.com/charitha-reddy1074/Neural-Style-Transfer.git
cd Neural-Style-Transfer
```

Install the required packages:

```bash
pip install tensorflow numpy pillow matplotlib gradio jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Neural_Style_Transfer.ipynb
```

Run the notebook cells in order.

## Usage

1. Open the notebook.
2. Load a content image.
3. Load a style image.
4. Configure the image size and optimization parameters.
5. Run the style-transfer optimization.
6. Monitor the content, style, and total loss values.
7. View the generated stylized image.
8. Use the Gradio interface for interactive experimentation.

## Important Hyperparameters

The following parameters influence the final result:

| Parameter | Description |
|---|---|
| `content_weight` | Controls how strongly the generated image preserves the content image |
| `style_weight` | Controls the strength of the artistic style |
| `learning_rate` | Controls the size of each optimization update |
| `iterations` | Number of optimization steps |
| `image_size` | Resolution used for processing |
| `style_layer_weights` | Relative importance of individual style layers |

### Hyperparameter Effects

- Increase `content_weight` to preserve more structure.
- Increase `style_weight` to produce a stronger artistic appearance.
- Increase the number of iterations for further refinement.
- Reduce image resolution for faster experimentation.
- Increase image resolution for better visual detail, at the cost of memory and processing time.

## Limitations

- Optimization can be computationally expensive.
- Results depend heavily on the selected content and style images.
- High-resolution images require significant memory.
- The generated image may contain color artifacts or unwanted textures.
- Excessive style weight can reduce content recognizability.
- The optimization process may require manual hyperparameter tuning.
- Output quality is subjective and cannot be fully represented by a single metric.

## Possible Improvements

Future improvements could include:

- Adding total variation loss to reduce image noise.
- Supporting arbitrary image resolutions.
- Preserving the original aspect ratio.
- Adding image previews and progress updates.
- Saving intermediate optimization results.
- Using mixed precision for faster computation.
- Implementing GPU acceleration.
- Adding batch evaluation with SSIM and FID.
- Using a feed-forward fast style-transfer network.
- Supporting multiple style images.
- Adding configurable style-layer weights.
- Creating a production-ready web deployment.

## Project Structure

```text
Neural-Style-Transfer/
│
├── Neural_Style_Transfer.ipynb
└── README.md
```

## Learning Outcomes

This project demonstrates practical knowledge of:

- Convolutional neural networks
- Transfer learning
- VGG19 feature extraction
- TensorFlow automatic differentiation
- Gradient-based image optimization
- Content and style loss design
- Gram matrix computation
- Image preprocessing and deprocessing
- Tensor manipulation
- Hyperparameter tuning
- Interactive machine learning interfaces
- Perceptual image evaluation

## Conclusion

This project demonstrates how a pre-trained CNN can be used as a perceptual feature extractor for image generation. Rather than training a new model, Neural Style Transfer optimizes image pixels to minimize the difference between the content and style representations extracted by VGG19.

The result is an artistic image that combines the semantic structure of one image with the visual appearance of another.
