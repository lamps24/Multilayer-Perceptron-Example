%%
% Implements the MLP Backward Propagation step
%
% The parameters received are:
% - X (N x D): Training datapoints matrix, where N is the 
% number of training data points, and D is the number of features
% - y_label (N x 1): True labels of each data point
% - Y_pred (N x K): Output of each output unit
% - Z (N x H+1): Matrix that contains the output from the hidden units,
% including the bias unit z0=+1
% - V (H+1 x K): Weights between each hidden unit and output unit
% - eta (1 x 1): The learning rate
%
% The function should return:
% - dW (D+1 x H): Updates for the weights in W
% - dV (H+1 x K): Updates for the weights in V
%
function [dW, dV] = BackwardPropagation(X, y_label, Y_pred, Z, V, eta)
%%%% YOUR CODE STARTS HERE
y_label_plus_one = y_label+1;
X2 = [ones(size(X,1),1) X]; %add column of ones to X matrix

%create r matrix of class labels
r = zeros(size(y_label_plus_one, 1), 10);
for i = 1:size(y_label_plus_one, 1)
    r(i, y_label_plus_one(i)) = 1;
end

%dV using 11.28
for h = 1:(size(Z,2))
    for i = 1:10
        dV(h, i) = eta * ( transpose(r(:,i) - Y_pred(:,i)) * Z(:,h) );
    end 
end

%dW
for h = 2:(size(Z,2))
    dW(:,h-1) = eta*(transpose((r - Y_pred)*transpose(V(h,:))) * (Z(:,h).*(1 - Z(:,h)).*X2)); 
end

%%%% 
end

