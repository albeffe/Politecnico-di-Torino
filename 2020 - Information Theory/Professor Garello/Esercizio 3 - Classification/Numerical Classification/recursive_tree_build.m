function [tree, labels] = recursive_tree_build(training_dataset, usable_features, tree, level, labels, current_path, last_label, numerical)
    
    % Condizione di terminazione
    % se tutte le labels sono uguali
    % se non ci sono piu attributi
    if size(usable_features, 2) == 0 || size(unique(training_dataset(:, 4)), 1) == 1
        % Padding
        if size(current_path, 2) < (size(training_dataset, 2)-1)
            for i=1:size(training_dataset, 2) - size(current_path, 2) -1
                blocco_fake.sign = 0;
                blocco_fake.val = -1;
                blocco_fake.feat = -1;
                current_path = [current_path blocco_fake];
            end
        end
        tree = [tree; current_path];
        labels = [labels last_label];
        return
    end
    
    % Calcolo entropia di ogni feature
    H_1 = entropy(transpose(training_dataset(:,1)));
    H_2 = entropy(transpose(training_dataset(:,2)));
    H_3 = entropy(transpose(training_dataset(:,3)));
    H_c = entropy(transpose(training_dataset(:,4)));
    H_vect = [H_1 H_2 H_3 H_c];
    
    % cerco la feature con max igr tra quelle rimaste
    IGR_max = 0;
    IGR = 0;
    IGR_num = 0;
    feature_max = 1;
    index = 1;
    threshold_max = 0;
    threshold = 0;
    flag = 0;
    
    for i=1:size(usable_features, 2)
        if numerical(usable_features(i)) == 1
            [IGR_num, threshold] = igr_calc_numerical(transpose(training_dataset(:,usable_features(i))), H_vect(4), training_dataset, usable_features(i));
        else
            IGR = igr_calc(transpose(training_dataset(:,usable_features(i))), H_vect(4), training_dataset, usable_features(i), H_vect(usable_features(i)));
        end
        
        if IGR > IGR_max
            IGR_max = IGR;
            feature_max = usable_features(i);
            index = i;
            flag = 0;
            IGR = 0;
        end
        if IGR_num >= IGR_max
            IGR_max = IGR_num;
            threshold_max = threshold;
            feature_max = usable_features(i);
            index = i;
            flag = 1;
            IGR_num = 0;
        end
    end
    usable_features(index) = [];
    %feature_max
    %IGR_max
    %threshold_max
    %usable_features
    
    % Parte ricorsiva se colonna selezionata non numerical
    if flag == 0
        valori_distinti_feature_max = unique(training_dataset(:, feature_max));
        for i=1:size(valori_distinti_feature_max, 1)
        
            % Partizionamento dataset
            splitted_dataset = [];
            for j=1:size(training_dataset, 1)
                if training_dataset(j, feature_max) == valori_distinti_feature_max(i)
                    splitted_dataset = [splitted_dataset; training_dataset(j, :)];
                end
            end
            
            blocchetto.sign = 0;
            blocchetto.val = valori_distinti_feature_max(i);
            blocchetto.feat = feature_max;
        
            % Calcolo most common label per valore distinto di feature
            last_label = mode(splitted_dataset(:, 4));
        
            % Ricorsione
            [tree, labels] = recursive_tree_build(splitted_dataset, usable_features, tree, level+1, labels, [current_path blocchetto], last_label, numerical);
        
        end
    % Parte ricorsiva se colonna selezionata è numerical
    else
        % Partizionamento sopra il threshold
        splitted_dataset = [];
        for j=1:size(training_dataset, 1)
            if training_dataset(j, feature_max) > threshold_max
                splitted_dataset = [splitted_dataset; training_dataset(j, :)];
            end
        end
        
        if size(splitted_dataset, 1) >= 1
            blocchetto.sign = 1;
            blocchetto.val = threshold_max;
            blocchetto.feat = feature_max;
        
            % Calcolo most common label per valore distinto di feature
            last_label = mode(splitted_dataset(:, 4));
        
            % Ricorsione
            [tree, labels] = recursive_tree_build(splitted_dataset, usable_features, tree, level+1, labels, [current_path blocchetto], last_label, numerical);
        
        end
        
        %%%%%%%%%%%%%
        
        % Partizionamento sotto il threshold
        splitted_dataset = [];
        for j=1:size(training_dataset, 1)
            if training_dataset(j, feature_max) <= threshold_max
                splitted_dataset = [splitted_dataset; training_dataset(j, :)];
            end
        end
        
        if size(splitted_dataset, 1) >= 1
            blocchetto.sign = -1;
            blocchetto.val = threshold_max;
            blocchetto.feat = feature_max;
        
            % Calcolo most common label per valore distinto di feature
            last_label = mode(splitted_dataset(:, 4));
        
            % Ricorsione
            [tree, labels] = recursive_tree_build(splitted_dataset, usable_features, tree, level+1, labels, [current_path blocchetto], last_label, numerical);
        end
    end
end