clc
clear all
close all

%H = - 10 * (0.2*log2(0.2) + 0.8*log2(0.8))

p = zeros(1,11);
for i=0:1023
    x = dec2bin(i,10)-'0';
    y = sum(x)+1;
    p(y) = p(y)+0.2^sum(x==0)*0.8^sum(x==1);
end
Hy = H(p)