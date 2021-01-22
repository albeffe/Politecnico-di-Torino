function [tree, labels] = classification_fit2(training_dataset)
    
    % Calcolo cardinalità alfabeti
    alfa_feature_cardinal_1 = size(unique(training_dataset(:,1)), 1);
    alfa_feature_cardinal_2 = size(unique(training_dataset(:,2)), 1);
    alfa_feature_cardinal_3 = size(unique(training_dataset(:,3)), 1);
    alfa_feature_cardinal_4 = size(unique(training_dataset(:,4)), 1);
    
    
    % Creazione albero
    usable_features = [1 2 3];
    tree = [];
    labels = [];
    current_path = [];
    last_label = -1;
    level = 1;
    [tree, labels] = recursive_tree_build(training_dataset, usable_features, tree, level, labels, current_path, last_label);
    labels = labels - 1;
end
