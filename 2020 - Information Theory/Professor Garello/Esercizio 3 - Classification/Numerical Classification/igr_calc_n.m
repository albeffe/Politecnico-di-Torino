function igr = igr_calc_n(array, H_c, dataset, attrib_num, H_x)
    
    H_c_x = 0;
    
    % istanzio vettore delle occorrenze
    occur = zeros(1, max(unique(array)));
    
    % riempimento vettore occorrenze
    for i=1:size(array, 2)
        occur(array(i)) = occur(array(i)) + 1;
    end
    
    % calcolo probabilità di ogni elemento del vettore
    for i=1:size(occur, 2)
        occur(i) = occur(i) / size(array, 2);
    end
    
    for i=1:size(unique(array), 2)
        new_array = [];
        
        for j = 1:size(dataset, 1)
            if dataset(j, attrib_num) == i
                new_array = [new_array dataset(j, 4)];
            end
        end
        
        H = entropy(new_array);
        H_c_x = H_c_x + (H * occur(i));
        
    end
    
    if H_x ~= 0
        igr = (H_c - H_c_x) / H_x;
    else
        igr = 0;
    end

end