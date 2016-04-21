function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
e = exp(1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
numCols = size(z,2);
numRows = size(z,1);

for col = 1:numCols
    for row= 1:numRows
        g(row,col) = 1 / (1 + e .^ (-1 * z));
    end 
end


% =============================================================

end
