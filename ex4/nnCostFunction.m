function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);
y = y_matrix;

%a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1c = ones(size(X, 1), 1);
a1 = [a1c X];
%X = 
%z2 equals the product of a1 and ?1
%disp(size(a1));
%disp(size(Theta1'));

z2 = a1 * Theta1';

%a2 is the result of passing z2 through g()
a2temp = sigmoid(z2);
%Then add a column of bias units to a2 (as the first column).
a2temp2 = ones(size(a2temp,1),1);
a2 = [a2temp2 a2temp];
%NOTE: Be sure you DON'T add the bias units as a new row of Theta.

%z3 equals the product of a2 and ?2
z3 = a2 * Theta2';

%a3 is the result of passing z3 through g()
a3 = sigmoid(z3);

innerterms = ((-1 * y) .* log(a3)) - ((1-y) .* log(1-a3));
Ksum = sum(innerterms);
iSum = sum(Ksum);
J_unreg = iSum / m;
%J = sum(sum(innerterms)) / m;
J_reg = 0;
%remove bias column from Theta1, Theta2
Theta1_nb =  Theta1(:,2:end);
Theta2_nb =  Theta2(:,2:end);
reg_innerterm_left = sum(sum(Theta1_nb .^ 2));
reg_innerterm_right = sum(sum(Theta2_nb .^ 2));
J_reg = (lambda / (2 * m)) * (reg_innerterm_left + reg_innerterm_right);
J = J_unreg + J_reg;

d3 = a3 - y_matrix;
d2 = (d3 * Theta2_nb) .* sigmoidGradient(z2);
Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1_grad_unreg = Delta1 / m;
Theta2_grad_unreg = Delta2 / m;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad_reg = (lambda / m) * Theta1;
Theta2_grad_reg = (lambda / m) * Theta2;


Theta1_grad = Theta1_grad_unreg + Theta1_grad_reg;
Theta2_grad = Theta2_grad_unreg + Theta2_grad_reg;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
