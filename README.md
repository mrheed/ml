Neural Network :
	https://cs231n.github.io/

Plan :
	https://cs231n.github.io/neural-networks-case-study/
	
# Machine Learning by Andrew Ng resume:
## Cost Function
- Linear function (hypothesis function) = h(x) = θ₀ + θ₁x
- Cost function = J(Θ₀, Θ₁) = 1/2m * sigma(h(xⁱ) - yⁱ)²
The idea of choosing the value of θ₀,θ₁ in linear function in order to get the accurate regression is to find the minimum value of J(Θ₀, Θ₁)

## Gradient Descent
Using gradient descent, we could find the minimum value of a function. 
Gradient descent definition (repeat until convergence):
 - Θₓ = Θₓ - ∝ * ∂(J(Θ₀, Θ₁)) / ∂Θₓ
		- The alpha represent as the learning rate of the algorithm and it controls how big the stepness
		- We have to update the value of Θ₀ and Θ₁ simultaneously like so
			- temp0 = Θ₀ - ∝ * ∂(J(Θ₀, Θ₁)) / ∂Θₓ
			- temp1 = Θ₁ - ∝ * ∂(J(Θ₀, Θ₁)) / ∂Θₓ
			- Θ₀ = temp0
			- Θ₁ = temp1
What if the Θₓ is already at the local minimum ?
 - The current value will be Θₓ - ∝ * 0, so one step of gradient descent does absolutely nothing
s we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease ∝ over time.
Each change of the cost function it will correspond to the hypothesis function value
Batch Gradient Descent using all the training examples to each step

## Linear Algebra
Matrix multiplication properties:
 - Comutative for scalar multiplication, but for matrix matrix, it doesnt (AB != BA)
 - Associative, given matrix A x B x C the first computation you can do is either A x B or B x C, because it will give the same result

Matrix Identity:
 - Denoted as I
 - The dimension is n x n
 - The identity matrix has the property 1 along the diagonal and 0 for others

Matrix Operation:
 - Matrix Inverse, if A is an m x m (square) matrix, and it if has an inverse
  - A(A⁻ⁱ) = A⁻ⁱA = I
	- A matrix that don't have an inverse are sometimes called the singular matrix or degenerate matrix (if the matrix is too close to 0)
 - Transpose, rotate the matrix 90 deg counter-clockwise and flip it horizontally

## Linear Regression
### Multiple Features

`  Size   |   nBed   |   nFlo   |   aHom   |   Price    `

`  2104   |     5    |     1    |    45    |   460      ` 

`	 1416   |     3    |     2    |    40    |   232      `

`	 1534   |     3    |     2    |    30    |   315      `

`	 852    |     2    |     1    |    36    |   178      `



Suppose we have training data like so, we are going to predict the price using variables size, nbed, nflo, and ahom.
Let's define the notation:
 - y  = price
 - x₀ = bias multiplier or maybe coeficient
 - x₁ = size
 - x₂ = nbed
 - x₃ = nflo
 - x₄ = ahom
 - n  = number of features
 - xⁱ = input(features) of the i-th training example 
 - xₓ = value of feature x in i-th training example

`	  	| x₀ |				| Θ₀ | `

`	  	| x₁ |				| Θ₁ | `

`	  	| x₂ |				| Θ₂ | `

` X =	| .. |		Θ =	| .. | `

`	  	| .. |				| .. | `

`	  	| xₓ |				| Θₓ | `

` h(x)	= Θ₀x₀ + Θ₁x₁ + Θ₂x₂ + ... + Θₓxₓ 
	  		= ΘᵀX `

The gradient descent computation is just like the notation above, derivative sigmoid(cost function) with respect of Θₓ over 2 * sum of training data
