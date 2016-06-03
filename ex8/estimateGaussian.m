function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

for i=1:n
    thisRow = X(:,i);
    mu(i,:) = mean(thisRow);
    sigma2(i,:) = (1 / m) * sum((thisRow - mu(i)).^ 2);
end

%mu = mean(X);
%muRow = mu';
%disp(muRow)

%mu = mean(X);
%size(X)
%size(mu)

%sigma2 = (1 / m) * sum((X' - mu').^ 2)';

%inner = X - mu;
%%sigma2 = (1 / m) * sum(inner .^ 2)';
%sigma2 = (1/m) * (X' * X);


 % mu2 = 1/m*sum(X)';
%size(mu2)
  
  %---for loop based sigma2---
  %for j=1:m
  %  temp = (X(j,:) - mu').^2;
  %  sigma2 += temp';
  %end
  %sigma2 = 1/m*sigma2

  %---vectorized sigma2---
 % sigma2 = 1/m*sum((X - mu').^2)';


% =============================================================


end
