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

%convert y vector to a matrix
%can also be done with "logical arrays"
sy = size(y,1);
Y = zeros(sy, num_labels);
Y = zeros(size(y,1), num_labels);
for i=1:sy,
	Y(i,y(i)) = 1;	% 5000	10
end;

%input_layer_size	= 400
%hidden_layer_size	= 25
%num_labels	= 10
%size(X) 	= 5000	400
%size(y) 	= 5000	1
%size(Y) 	= 5000	10
%lambda 	= 1 or 0

%a1 = [ones(size(X, 1), 1) X];				% 5000	401
%Theta1;									% 25		401
%z2 = a1 * Theta1';							% 5000	25
%a2 = [ones(size(z2, 1), 1) sigmoid(z2)];	% 5000	26
%Theta2;									% 10		26
%z3 = a2 * Theta2';							% 5000	10
%a3 = sigmoid(z3);							% 5000	10

%forward prop in one line
a3 = sigmoid([ones(size([ones(size(X, 1), 1) X] * Theta1', 1), 1) sigmoid([ones(size(X, 1), 1) X] * Theta1')] * Theta2');

for i = 1:m,
	J = J + 1 / m * (-Y(i,:) * log(a3(i,:))' - (1 - Y(i,:)) * log(1 - a3(i,:))');
end;

regTheta1 = 0;
for j=1:(hidden_layer_size),
	for k=2:(input_layer_size+1),
		regTheta1 = regTheta1 + Theta1(j, k)^2;
	end;
end;
regTheta2 = 0;
for j=1:(num_labels),
	for k=2:(hidden_layer_size+1),
		regTheta2 = regTheta2 + Theta2(j, k)^2;
	end;
end;

regFactor = lambda / (2 * m) * (regTheta1 + regTheta2);

J = J + regFactor;


% -------------------------------------------------------------

Delta1 = 0;
Delta2 = 0;
for t=1:m,
	%STEP 1
	a1 = [1 X(t,:)];			% 1		401
	Theta1;						% 25		401
	z2 = a1 * Theta1';		% 1		25
	a2 = [1 sigmoid(z2)];	% 1		26
	Theta2;						% 10		26
	z3 = a2 * Theta2';		% 1		10
	a3 = sigmoid(z3);		% 1		10

	%STEP 2
	delta3 = a3 - (1:num_labels == y(t));					% 1	10
	%STEP 3
	delta2 = delta3 * Theta2 .* [1 sigmoidGradient(z2)];	% 1	26
	%STEP 4
	Delta1 = Delta1 + delta2(2:end)' * a1;	% 25	401
	Delta2 = Delta2 + delta3' * a2;			% 10	26
end;

%STEP 5
Theta1_grad = (1/m) * Delta1; % 25	401
Theta2_grad = (1/m) * Delta2; % 10	26

%Regularized
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
