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

%inc = [.01 .03 .1 .3 1 3 10 30];
%bestError = y(1)^2 * 10;
%for i = inc,
%	for j = inc,
%		thisC = i;
%		thisSigma = j;
%		model = svmTrain(X, y, thisC, @(x1, x2) gaussianKernel(x1, x2, thisSigma));
%		predictions = svmPredict(model, Xval);
%		thisError = mean(double(predictions ~= yval));
%		fprintf('thisC = %f, thisSigma = %f, thisError = %f\n', thisC, thisSigma, thisError);
%		if bestError > thisError,
%			bestError = thisError;
%			C = thisC;
%			sigma = thisSigma;
%			fprintf('Better! C = %f, sigma = %f, bestError = %f\n', C, sigma, bestError);
%		end;
%	end;
%end;
%fprintf('Here is the best:\n   C = %f, sigma = %f, bestError = %f\n', C, sigma, bestError);

C = 1;
sigma = .1;

% =========================================================================

end
