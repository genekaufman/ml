function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
disp('12 size(J):');
disp(size(J));

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
Term1 = 0;
Term2 = 0;
for i = 1:m
    h_theta_x = sigmoid(theta' .* X(i,:));
    fprintf('%d:',i);
    disp('htx:')
    disp(h_theta_x);
    Term1 = Term1 - y(i) * log(h_theta_x );
    Term2 = Term2 + (1 - y(i)) * log(1 - h_theta_x);
    %grad(i) = (sigmoid(theta' * X(i)) - y(i)) * X(i);
    %h = X * theta;

end
JTemp = (Term1 - Term2) / m;
disp('39 size(JTemp):');
disp(size(JTemp));
disp(JTemp);
J = JTemp(1);
disp('42 size(J):');
disp(size(J));

%h_theta_x = sigmoid(theta' * X);
%Term1 = (-1 * y) * log(h_theta_x);




% =============================================================

end
