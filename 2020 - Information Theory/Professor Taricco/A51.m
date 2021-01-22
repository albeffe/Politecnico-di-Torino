function [newstate1, newstate2, newstate3, out] = A51(arr1, arr2, arr3)
    maj = mode([arr1(8) arr2(10) arr3(10)])
    
    con1 = [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1];
    con2 = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1];
    con2 = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1];
    
    if arr1(8) == maj
        [newstate1, out1] = shift_reg(arr1, con1);
    end
    
    if arr2(10) == maj
        [newstate2, out2] = shift_reg(arr2, con2);
    end
    
    if arr3(10) == maj
        [newstate3, out3] = shift_reg(arr3, con3);
    end

    out = mod(sum([out1, out2, out3]),2);
end