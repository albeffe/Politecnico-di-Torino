clc
clear all
close all

% Training dataset

% Temp mapping
% Normal = 1
% High = 2
% Very_High = 3

% Training dataset
v1 = [30 1 10 1];
v2 = [30 1 70 1];
v3 = [30 2 20 1];
v4 = [30 2 80 2];
v5 = [60 1 40 1];
v6 = [60 1 60 2];
v7 = [60 2 50 1];
v8 = [60 2 60 2];

training_dataset = [v1; v2; v3; v4; v5; v6; v7; v8];
numerical = [1 0 1 0];

% Test Dataset
test_dataset = [31 1 49;
                31 2 51;
                29 2 51;
                29 1 51];

% Fit (Creazione Albero in base al Training dataset)
[tree, labels] = classification_fit2(training_dataset, numerical);

% Predict (Classificazione Test dataset)
class_array = [];
for array=1:size(test_dataset, 1)
    class = classification_predict2(test_dataset(array, :), tree, labels);
    class_array = [class_array class];
end

disp('Test dataset classes:')
disp(class_array)
