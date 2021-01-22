function H = entropy(array)
    
    H = 0;
    occur = zeros(1, max(unique(array)));
    
    for i=1:size(array, 2)
        occur(array(i)) = occur(array(i)) + 1;
    end
    
    % calcolo probabilità di ogni elemento del vettore
    for i=1:size(occur, 2)
        occur(i) = occur(i) / size(array, 2);
    end
    
    % calcolo entropia
    for i=1:size(occur, 2)
        if occur(i) ~= 0
            H = H + occur(i)*log2(1/occur(i));
        end 
    end
    
    if size(occur, 2) == 1
        H = 0;
    end

end