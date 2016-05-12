function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


hh = X * theta - y;
unregJ = (1/(2*m)) * (hh' * hh);

unreg_grad = (X' * hh) / m;

theta(1) = 0;

regJ = (theta' * theta) * (lambda/(2*m));
J = unregJ + regJ;


%Theta1_nb =  theta(:,2:end);
%The regularized gradient term is theta scaled by (lambda / m). 
%Again, since theta(1) has been set to zero, it does not contribute to the 
%regularization term.
grad_reg= theta * (lambda/m);
%disp(grad_reg)
grad = unreg_grad + grad_reg;




% =========================================================================

grad = grad(:);

end
