clc
clear all
close all

P = zeros(8);
for i=1:8
    s1 = dec2bin(i-1,3)-'0';
    for j=1:8
        s2 = dec2bin(j-1,3)-'0';
        if all(s1(2:3)==s2(1:2))
            if sum(s1)<2
                if s2(3)==0,p=0.2; else,p=0.8;end
            else
                p=0.5;
            end
            P(i,j) = p;
        end
    end
end

% Analizziamo ora la state evolution

q = [1 zeros(1,7)];
for i=1:10
    q = q*P;
    fprintf('%4d',i);
end

A = P'-eye(8);
A(1,:) = ones(1,8);
q = A\[1 zeros(1,7)]';

H = sum(q([1 2 3 5])) * Hb(0.2)+sum(q([4 6 7 8]))*Hb(0.5);




