function [igr_max, threshold] = igr_calc_numerical(array, H_c, dataset, attrib_num)  
    
    sort_array = sort(array);
    igr_max = 0;
    threshold = 0;
    
    for i=1:size(sort_array, 2)
        converted_array = [];
        
        % Rimappo il vettore
        for j=1:size(sort_array, 2)
            if array(j) > sort_array(i)
                converted_array = [converted_array 2];
            else
                converted_array = [converted_array 1];
            end
        end
        
        % Calcolo igr
        tmp_dataset = dataset;
        tmp_dataset(:, attrib_num) = converted_array;
        H_x = entropy(converted_array);
        IGR = igr_calc_n(converted_array, H_c, tmp_dataset, attrib_num, H_x);
        
        if IGR > igr_max
            igr_max = IGR;
            threshold = sort_array(i);
        end
        
    end
   
end