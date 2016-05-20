function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_range = [0.01, 0.035, 0.1, 0.35, 1, 3.5, 10, 35];
sigma_range = C_range;
bestError = 0;
vals2Use = [C, sigma];
firstRun = 1;
x1 = [1 2 1]; x2 = [0 4 -1];
for thisC = 1:length(C_range)
    for thisSigma = 1:length(sigma_range)
        
        model= svmTrain(X, y, C_range(thisC), @(x1, x2) gaussianKernel(x1, x2,  sigma_range(thisSigma)));
        pred = svmPredict(model, Xval);
        CS = [C_range(thisC), sigma_range(thisSigma)];
        error =  mean(double(pred ~= yval));
        betterResults = 0;
        if firstRun == 1 
            betterResults = 1;
        elseif error < bestError
            betterResults = 1;
        end
        
        if betterResults == 1
            bestError = error;
            vals2Use = CS;
            fprintf(['49 [%0.5f %0.5f] pred: %0.5f\n'],C_range(thisC), sigma_range(thisSigma),pred);
        end
        firstRun = 0;
    end
end
C = vals2Use(1);
sigma = vals2Use(2);


% =========================================================================

end
