function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% C(k) is the count of examples assigned to centroid k
C = histc(idx, 1:K);

% iterate over all features (this homework only has 2; x and y)
for j = 1:n,
	% for every feature accumarray sums up all values for that feature, then 
	% divides by the count to get the mean. This mean is then saved into the 
	% correct centroid feature column.
	centroids(:, j) = accumarray(idx, X(:, j)) ./ C; 
end;

%Code below only works for 2 dimensional X datasets
%[sumX] = accumarray(idx, X(:, 1));
%[sumY] = accumarray(idx, X(:, 2));
%for k = 1:K,
%	C(k)  = histc(idx, k);
%	centroids(k, :) = [sumX(k) / C(k) sumY(k) / C(k)];
%end;


% =============================================================


end

