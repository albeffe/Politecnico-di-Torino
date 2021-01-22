clc
clear all
close all

P = zeros(10);
for x=1:10
    for y=1:10
        P(x,y) = 2*x+5*y;
    end
end

K = sum(P(:));
Pxy = P/K;
Px = sum(Pxy, 2);
Py = sum(Pxy, 1);
Hxy = H(Pxy(:));
Hx = H(Px(:));
Hy = H(Py(:));
Hx_y = Hxy-Hy;
Hy_x = Hxy-Hx;
Ixy = Hx+Hy-Hxy;
Pz = zeros(1,20);
for x=1:10
    for y=1:10
        Pz(x+y) = Pz(x+y) + Pxy(x,y);
    end
end
Hz = H(Pz(:));