%% 4a) 

% read data: call ReadNormalizedOptdigitsDataset
[X_trn_norm, y_trn, X_val_norm, y_val, X_tst_norm, y_tst] = ReadNormalizedOptdigitsDataset("C:\Users\lamps\Documents\Class\CSCI 5521 - Machine Learning\HW3\Question4\optdigits_train.txt", "C:\Users\lamps\Documents\Class\CSCI 5521 - Machine Learning\HW3\Question4\optdigits_valid.txt", "C:\Users\lamps\Documents\Class\CSCI 5521 - Machine Learning\HW3\Question4\optdigits_test.txt");

Hs = [4,8,12,16,20,24];
training_error = zeros(length(Hs),1);
validation_error = zeros(length(Hs),1);

% check training and validation error for each option of H
for i=1:length(Hs)
    H = Hs(i);

    % train MLP using current H using MLPTrain
    [Y_trn_pred,Z,W,V] = MLPTrain(X_trn_norm, y_trn, H);

    % calculate error rate for Y predicted to the training set using CalculateErrorRate
    training_error(i) = CalculateErrorRate(Y_trn_pred, y_trn);

    fprintf('Training set error rate when H=%d: %f\n', H, training_error(i));
    
    % Predict Y for the validation set, using ForwardPropagation
    [Y_val_pred,Z] = ForwardPropagation(X_val_norm, W, V);

    % calculate error rate for Y predicted to the validation set using CalculateErrorRate
    validation_error(i) = CalculateErrorRate(Y_val_pred, y_val);
    
    fprintf('Validation set error rate when H=%d: %f\n', H, validation_error(i));
    
end

% Plot training and validation error using PlotTrainingValidationError
PlotTrainingValidationError(Hs,training_error, validation_error);
 
% train MLP using the best number of hidden units MLPTrain
[Y_trn_pred,Z,W,V] = MLPTrain(X_trn_norm, y_trn, 20);

% Predict Y for the test set, using ForwardPropagation
[Y_tst_pred,Z] = ForwardPropagation(X_tst_norm, W, V);

% calculate error rate for Y predicted to the test set using CalculateErrorRate
test_error = CalculateErrorRate(Y_tst_pred, y_tst);

fprintf('Test set error rate when H=%d: %f\n', Hs(5), test_error);

% Train the MLP with 2 hidden units, using MLPTrain
[Y_trn_pred,Z_trn,W,V] = MLPTrain(X_trn_norm, y_trn, 2);

% Predict Y for the validation and test set, using ForwardPropagation
[Y_val_pred,Z_val] = ForwardPropagation(X_val_norm, W, V);
[Y_tst_pred,Z_tst] = ForwardPropagation(X_tst_norm, W, V);

% Do a 2D scatter showing Z for the training, validation and test set in
% separate figures, using PlotZ2DScatter for each figure
PlotZ2DScatter(Z_trn,Y_trn_pred)
PlotZ2DScatter(Z_val,Y_val_pred)
PlotZ2DScatter(Z_tst,Y_tst_pred)


% Train the MLP with 3 hidden units, using MLPTrain
[Y_trn_pred,Z_trn,W,V] = MLPTrain(X_trn_norm, y_trn, 3);

% Predict Y for the validation and test set, using ForwardPropagation
[Y_val_pred,Z_val] = ForwardPropagation(X_val_norm, W, V);
[Y_tst_pred,Z_tst] = ForwardPropagation(X_tst_norm, W, V);

% Do a 3D scatter showing Z for the training, validation and test set in
% separate figures, using PlotZ3DScatter for each figure
PlotZ3DScatter(Z_trn,Y_trn_pred)
PlotZ3DScatter(Z_val,Y_val_pred)
PlotZ3DScatter(Z_tst,Y_tst_pred)
