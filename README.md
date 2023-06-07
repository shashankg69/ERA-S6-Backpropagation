# Backpropagation

Backpropagation is a key technique used in training neural networks through gradient descent. It involves propagating the error backward through the network to update the weights and biases. 
  - Neural networks consist of interconnected layers of nodes (neurons) that process input data to produce output predictions.
  - During the forward pass, input data is fed through the network, and activations are computed for each neuron using an activation function.
  - The predicted output is compared to the desired output using a loss function.
  - Backpropagation calculates the gradient of the error with respect to each weight and bias in the network.
  - The gradient descent algorithm is then used to update the weights and biases in the opposite direction of the gradient, gradually reducing the error.
  - The process of forward pass, error calculation, and backward pass is repeated for multiple iterations (epochs) until the model converges or reaches a stopping criterion.

## Calculating Backpropagation:
   ![](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/85ee679d-1edc-4029-a8db-38278aa0b720)
   
   - `i1` and `i2` : Neurons of the input Layer
   - `h1` and `h2` : Weighted sum of the input neurons
   
     - `h1 = w1*i1 + w2*i2`
     - `h2 = w3*i1 + w4*i2`
     
   - `a_h1` and `a_h2` : When an activation function σ (sigmoid function) is applied to h1 and h2, it results in `a_h1` and `a_h2`
   
      - The formula for calculating the Sigmoid Function of a value is : `σ(x) = 1 / (1 + exp(-x))`
      - So, by using the above formula `a_h1 = σ(h1) = 1/(1 + exp(-h1))` and `a_h2 = σ(h2) = 1/(1 + exp(-h2))`
      
   - Similarly, the neurons for the output layer can be calculated
   
      -  `o1 = w5*a_h1 + w6*a_h2`
      -  `o2 = w7*a_h1 + w8*a_h2`
      -  `a_o1 = σ(o1) = 1/(1 + exp(-o1))`
      -  `a_o2 = σ(o2) = 1/(1 + exp(-o2))`
      
### Backpropagation:
   - The total loss `E_total` is defined as the sum of the individual losses `E1` and `E2` 
      
      - `E_total`=`E1`+`E2`
      - `E1`=`½ * (t1 - a_o1)²`
      - `E2`=`½ * (t2 - a_o2)²`
   - The partial derivative of`E_total`with respect to each weight (e.g., w5, w6, w7, w8, w1, w2, w3, w4) is computed to determine how each weight affects the overall error.

