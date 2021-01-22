function  C=Hc(p)
    N=numel(p);
    C=cell(1,N);
    if numel(p) == 2
       C{1} = 0;
       C{2} = 1;
    else
        [p, idx] = sort(p);
        q = p(1) + p(2);
        %Find the code corresponding to [p(1)+p(2)+p(3:end)]
        %by recursive call to HC
        C1 = Hc([p(1)+p(2)+p(3:end)]);
        %Merge this result with the 2 pieces '0' and '1'
        C{idx(1)} = [C1{1},0];
        C{idx(2)} = [C1{1},1];
        for i=3:N
            C{idx(i)} = C1{i-1}
        end
    end