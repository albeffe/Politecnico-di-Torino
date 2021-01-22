function class_array = classification_predict2(array, tree, labels)
    cnt = 0;
    for i=1:size(tree, 1)
        for j=1:size(tree, 2)
            
            if cnt == size(tree, 2) || ((tree(i, j).feat) == -1)
                if cnt == size(tree, 2)
                    class_array = labels(i-1);
                else
                    class_array = labels(i);
                end
                return
            end
            
            if tree(i, j).sign == 0
                if array(tree(i, j).feat) == tree(i, j).val
                    cnt = cnt + 1;
                end
            
                if (array(tree(i, j).feat) ~= tree(i, j).val)
                    cnt = 0;
                    break
                end
            else
                if tree(i, j).sign == 1
                    if array(tree(i, j).feat) > tree(i, j).val
                        cnt = cnt + 1;
                    end
            
                    if (array(tree(i, j).feat) <= tree(i, j).val)
                        cnt = 0;
                        break
                    end
                else
                    if array(tree(i, j).feat) <= tree(i, j).val
                        cnt = cnt + 1;
                    end
            
                    if (array(tree(i, j).feat) > tree(i, j).val)
                        cnt = 0;
                        break
                    end
                end
            end
        end
    end
    
end
