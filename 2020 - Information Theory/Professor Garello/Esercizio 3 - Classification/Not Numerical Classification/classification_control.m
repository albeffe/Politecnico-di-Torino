% clc
% clear all
% close all

% Training dataset

% Temp mapping
% Normal = 1
% High = 2
% Very_High = 3

v1 = [2 2 1 2];
v2 = [3 2 2 2];
v3 = [1 1 1 1];
v4 = [2 2 2 2];
v5 = [2 1 2 1];
v6 = [1 2 1 1];
v7 = [1 1 2 1];

training_dataset = [v1; v2; v3; v4; v5; v6; v7];

% fit
[tree, labels] = classification_fit2(training_dataset);

% predict
test_dataset = [1 1 1;
                2 2 2;
                3 1 2];

class_array = [];
for array=1:size(test_dataset, 1)
    class = classification_predict2(test_dataset(array, :), tree, labels);
    class_array = [class_array class];
end

disp(class_array)
