function n = AvgBits(p)

    if numel(p) == 2
        n=1;
    else
        p = sort(p);
        q = p(1) + p(2);
        n = q + AvgBits([q, p(3:end)]);
    end
  
    