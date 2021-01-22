function H = Hb(p)
if p == 0
    H = 0;
else
    H = -p*log2(p)-(1-p)*log2(1-p);
end