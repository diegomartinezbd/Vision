# Deep Learning Terms Dictionary

## 1. **ReLU (Rectified Linear Unit)**
A popular activation function used in neural networks. It outputs the input directly if it is positive, and zero otherwise.

### Formula:
\[
\text{ReLU}(x) = \max(0, x)
\]

### Characteristics:
- Introduces non-linearity to the network.
- Helps mitigate the vanishing gradient problem.
- Computationally efficient.

---

## 2. **ReLU6**
A variant of ReLU that caps the output at 6, providing numerical stability and suitability for low-precision computations.

### Formula:
\[
\text{ReLU6}(x) = \min(\max(0, x), 6)
\]

### Characteristics:
- Used in mobile-friendly architectures like MobileNet.
- Prevents exploding activations and improves quantization.

---

## 3. **Global Average Pooling (GAP)**
A pooling technique that reduces each feature map to a single value by taking the spatial average.

### Formula:
For a feature map \( X \) of size \( H \times W \) and channel \( c \):
\[
GAP_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{i,j,c}
\]

### Characteristics:
- Reduces dimensions without introducing additional parameters.
- Helps prevent overfitting.
- Commonly used in architectures like ResNet and MobileNet.

---

## 4. **Depthwise Separable Convolutions**
An efficient convolutional operation that splits a standard convolution into two steps:
1. **Depthwise Convolution:** Applies a single filter per input channel.
2. **Pointwise Convolution:** Combines the outputs of the depthwise convolution using a \( 1 \times 1 \) kernel.

### Characteristics:
- Significantly reduces computational cost compared to standard convolutions.
- Used in MobileNet for lightweight model design.

---

## 5. **Vanishing Gradient Problem**
A problem in deep neural networks where gradients become very small during backpropagation, making it difficult for the network to learn.

### Cause:
- Activation functions like sigmoid or tanh squash input into small ranges, leading to diminishing gradients.

### Solution:
- Use activation functions like ReLU or architectures like ResNet with skip connections.

---

## 6. **Skip Connections**
A technique introduced in ResNet to mitigate the vanishing gradient problem. It allows gradients to flow directly through the network by skipping certain layers.

### Formula:
For an input \( x \) to a layer:
\[
y = F(x) + x
\]
Where \( F(x) \) represents the transformation applied by the skipped layer.

### Characteristics:
- Enables training of very deep networks.
- Helps in learning identity mappings.

---

## 7. **Width Multiplier (α)**
A hyperparameter used in MobileNet to control the number of filters in each layer.

### Formula:
\[
\text{Number of Filters in Layer} = \alpha \times \text{Original Filters}
\]

### Characteristics:
- Reduces model size and computational cost.
- Common values: \( \alpha = 0.75, 0.5 \), etc.

---

## 8. **Resolution Multiplier (ρ)**
A hyperparameter in MobileNet that adjusts the input image resolution.

### Formula:
\[
\text{Input Size} = \rho \times \text{Original Size}
\]

### Characteristics:
- Trades accuracy for computational efficiency.
- Common values: \( \rho = 1.0, 0.5 \), etc.

---

## 9. **Fully Connected (FC) Layer**
A layer where every neuron is connected to every neuron in the previous layer.

### Formula:
For input \( x \) with weights \( W \) and bias \( b \):
\[
y = Wx + b
\]

### Characteristics:
- Typically used for classification tasks.
- High parameter count can lead to overfitting.

---

## 10. **Softmax**
An activation function commonly used in the output layer for multi-class classification. It converts logits into probabilities.

### Formula:
For an input vector \( z \):
\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

### Characteristics:
- Ensures outputs are in the range \( [0, 1] \) and sum to \( 1 \).
- Suitable for multi-class classification problems.

---

## 11. **Dropout**
A regularization technique that randomly disables neurons during training to prevent overfitting.

### Formula:
During training, a neuron \( i \) is retained with probability \( p \):
\[
y_i = x_i \cdot \text{mask}_i
\]
Where \( \text{mask}_i \sim \text{Bernoulli}(p) \).

### Characteristics:
- Reduces overfitting by promoting redundancy in the network.
- Commonly used in AlexNet and VGG.

---

## 12. **Batch Normalization**
A technique to normalize the inputs to each layer, improving training stability and convergence speed.

### Formula:
For a mini-batch with mean \( \mu \) and variance \( \sigma^2 \):
\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]
The normalized input is then scaled and shifted:
\[
y = \gamma \hat{x} + \beta
\]

### Characteristics:
- Reduces internal covariate shift.
- Enables higher learning rates.

