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
      
      - `E_total` = `E1` + `E2`
      - `E1` = `½ * (t1 - a_o1)²`
      - `E2` = `½ * (t2 - a_o2)²`
      
   - The partial derivative of `E_total` with respect to each weight (e.g., `w5`, `w6`, `w7`, `w8`, `w1`, `w2`, `w3`, `w4`) is computed to determine how each weight affects the overall error.
   
   - The derivatives of `E_total` with respect to the weights in the output layer (`w5`, `w6`, `w7`, `w8`), the chain rule is applied to calculate these derivatives. This involves considering the derivative of the activation function        (σ) and the derivative of the weighted sum (`o1` and `o2`) with respect to each weight.
   
      - The expression `∂E_total/∂w5`, represents the partial derivative of the total error `E_total` with respect to the weight `w5`.

      - The total error `E_total` is defined as the sum of two individual errors, `E1` and `E2`. Therefore, we can rewrite `∂E_total/∂w5` as `∂(E1 + E2)/∂w5`.

      - Since we are interested in `∂E_total/∂w5`, we can focus on the first term `E1`. Thus, `∂(E1 + E2)/∂w5` becomes `∂E1/∂w5`.

      - Now, we need to calculate `∂E1/∂w5` using the chain rule. According to the given formula, `∂E1/∂w5` = `∂E1/∂a_o1` * `∂a_o1/∂o1` * `∂o1/∂w5`.

      - The first part, `∂E1/∂a_o1`, represents the partial derivative of the error `E1` with respect to the output activation `a_o1`. In this case, `E1` is defined as `½ * (t1 - a_o1)²`, where `t1` represents the target output for the         first output neuron. Taking the derivative with respect to `a_o1` gives `(a_o1 - t1)`.

      - The second part, `∂a_o1/∂o1`, represents the partial derivative of the sigmoid activation function `σ(o1)` with respect to the weighted sum `o1`. By taking the derivative of the sigmoid function, we obtain `∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1).`

      - The third part, `∂o1/∂w5`, represents the partial derivative of the weighted sum `o1` with respect to the weight `w5`. In this case, `o1` is calculated as `w5` * `a_h1` + `w6` * `a_h2`, and since `w5` is directly multiplied by           `a_h1`, the derivative is simply `a_h1`.
      
      - By combining these partial derivatives, we arrive at the final expression for `∂E_total/∂w5`: `(a_o1 - t1)` * `a_o1` * `(1 - a_o1)` * `a_h1`. 
      
      - So, similarly we can calculate the other partial derivates 
        			
        - ∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2					
        - ∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1					
        - ∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2					 

  - The derivatives of `E_total` with respect to the weights in the hidden layer (`w1`, `w2`, `w3`, `w4`) are also computed using the chain rule. These derivatives depend on the derivatives of `E_total` with respect to the activations       of the hidden layer neurons (`a_h1` and `a_h2`).
  
       - Starting with `∂E1/∂a_h1`, this represents the partial derivative of the error `E1` with respect to the hidden neuron activation `a_h1`.

       - According to the given formula, `E1` = `½ * (t1 - a_o1)²`, where `t1` is the target output for the first output neuron and `a_o1` is the actual output of the first output neuron. Taking the derivative of `E1` with respect to            a_h1 involves the chain rule. The partial derivative is calculated as follows: `∂E1/∂a_h1` = `∂E1/∂a_o1` * `∂a_o1/∂o1` * `∂o1/∂a_h1`.

       - `∂E1/∂a_o1` represents the partial derivative of `E1` with respect to the actual output `a_o1`. The derivative of `½ * (t1 - a_o1)²` with respect to `a_o1` is `(a_o1 - t1)`.

       - `∂a_o1/∂o1` represents the partial derivative of the sigmoid activation function `σ(o1)` with respect to the weighted sum `o1`. The derivative of the sigmoid function is given by `∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)`.

       - `∂o1/∂a_h1` represents the partial derivative of the weighted sum `o1` with respect to the activation `a_h1`. In this case, o1 is calculated as `w5 * a_h1 + w6 * a_h2`, and since `a_h1` is directly multiplied by `w5`, the                 derivative is simply  `w5`.
       
       - By combining these partial derivatives, we arrive at the expression `∂E_total/∂w1` = (`(a_01 - t1)` * `a_o1` * `(1 - a_o1)` * `w5` +  `(a_02 - t2)` * `a_o2` * `(1 - a_o2)` * `w7`) * `a_h1` * `(1 - a_h1)` * `i1`												 
        
       - Similarly , by calculating the other partial derivates 
       
          -  ∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2												
          - ∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1												
          - ∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2												
  
  - Now by substituting all the initial values we get
  - ![image](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/b1edb9c1-5f46-46a5-b324-013901922144)
    
    - To update the weights in the backpropagation algorithm, we use gradient descent. The formula to update the weights is as follows:
      
      - `new_w1 = old_w1 - learning_rate * ∂E_total/∂w1` 
    
    - Similarly, we can update the other weights (`w2`, `w3`, `w4`, `w5`, `w6`, `w7`, `w8`) using their respective gradients and the same formula.
    - The learning rate (n) determines the step size of each weight update. 
    
   ## This is how the network reacts to different learning Rates:
   
   ### Learning Rate : 0.1
   ![0 1](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/f583be07-bbec-40af-9b53-fcd7ba5b4bbc)
   
   ### Learning Rate : 0.2
   ![0 2](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/53ebcd34-7e84-4669-bdc2-68c7bdfa5d5d)
   
   ### Learning Rate : 0.5  
   ![0 5](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/06dd2819-5dd4-424d-8f33-4320c4fb2180)
   
   ### Learning Rate : 0.8  
   ![0 8](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/2e0adb5c-7edb-434a-b7ec-8ed5e0cf4589)
   
   ### Learning Rate : 1.0
   ![1 0](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/89c79266-7f86-456f-bf8a-d8deaa041336)
   
   ### Learning Rate : 2.0
   ![2 0](https://github.com/Shashank-Gottumukkala/ERA-S6/assets/59787210/f14f5143-b340-4fd3-b508-5e6d9cc2bc61)
