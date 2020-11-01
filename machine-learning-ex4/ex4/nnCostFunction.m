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

% ====================== PART 1 - FORWARD PROPOGATION ======================
% layer 1
a_1 = [ones(m, 1) X];       % m x features+1) = 5000 x 401

% layer 2
z_2 = a_1 * Theta1';        % (m x features+1) (features+1 x hidden_layer_size) = 5000 x 25
a_2 = sigmoid(z_2);         % m x hidden_layer_size =  5000 x 25
a_2 = [ones(m, 1) a_2];     % m x hidden_layer_size+1 =  5000 x 26

% layer 3
z_3 = a_2 * Theta2';        % (m x hidden_layer_size+1) (hidden_layer_size+1 x num_labels) =  5000 x 10
a_3 = sigmoid(z_3);         % m x num_labels =  5000 x 10
h = a_3;                    % m x num_labels =  5000 x 10

Y = (1:num_labels) == y;    % m x num_labels == 5000 x 10

J = - 1/m * sum(sum(( Y .* log(h) + (1-Y) .* log(1-h))));   % sum of 5000 x 10
% -------------------------------------------------------------



% ====================== PART 2 - BACK PROPOGATION ======================
% for i=1:m % iterate for every sample

%     % layer 1
%     ai_1 = a_1(i,:);           % 1 x 401

%     % layer 2
%     zi_2 = z_2(i,:);           % 1 x 25
%     ai_2 = a_2(i,:);           % 1 x 26

%     % layer 3
%     zi_3 = z_3(i,:);           % 1 x 10
%     ai_3 = a_3(i,:);           % 1 x 10

%     Yi = y(i,:);               % 1 x 10

%     delta_3 = ai_3 - Yi;       % 1 x 10

%     delta_2 = (delta_3 * Theta2) .* [ 1 sigmoidGradient(zi_2)];   % (1 x 10) (10 x 26) = 1 x 26

%     delta_2 = delta_2(2:end);   % 1 x 25

%     Theta1_grad += (delta_2' * ai_1); % 25 x 401
%     Theta2_grad += (delta_3' * ai_2); % 10 x 26

% end;

% Theta1_grad /= m; % 25 x 401
% Theta2_grad /= m; % 10 x 26

delta_3 = a_3 - Y;       % 5000 x 10
delta_2 = (delta_3 * Theta2) .* [ ones(m,1) sigmoidGradient(z_2)];   % (5000x x 10) (10 x 26) = 5000 x 26
delta_2 = delta_2(:, 2:end);   % 5000 x 25

Theta1_grad = (1/m) * (delta_2' * a_1); % 25 x 401
Theta2_grad = (1/m) * (delta_3' * a_2); % 10 x 26

% -------------------------------------------------------------

% ====================== PART 3 - Regularize ======================

J += (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));

Theta1_grad += (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad += (lambda/m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
