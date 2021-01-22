function class_array = classification_predict2(array, tree, labels)
    cnt = 1;
    
    for i=1:size(tree, 1)
        for j=1:size(tree, 2)
            if cnt == size(tree, 2) || ((tree(i, j).feat) == -1)
                class_array = labels(i);
                return
            end
            
            if array(tree(i, j).feat) == tree(i, j).val
                cnt = cnt + 1;
            end
            
            if (array(tree(i, j).feat) ~= tree(i, j).val)
                cnt = 1;
                break
            end
        end
    end
    
end
