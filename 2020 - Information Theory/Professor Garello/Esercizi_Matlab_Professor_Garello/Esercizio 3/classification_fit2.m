function [tree, labels] = classification_fit2(training_dataset, numerical)    
    
    % Creazione albero
    usable_features = [1 2 3];
    tree = [];
    labels = [];
    current_path = [];
    last_label = -1;
    level = 1;
    [tree, labels] = recursive_tree_build(training_dataset, usable_features, tree, level, labels, current_path, last_label, numerical);
    labels = labels - 1;
    
end
