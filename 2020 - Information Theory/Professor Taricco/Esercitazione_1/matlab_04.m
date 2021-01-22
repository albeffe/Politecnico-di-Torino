clc
clear all
close all

P = zeros(10);
for b=1:10
    for a=1:10
        P(b,a) = a+b;
    end
end
K = sum(P,2);
for b=1:10
    for a=1:10
        P(b,a) = P(b,a)/K(b);
    end
end
q = [1 zeros(1,9)];
for i=1:6
    q = q*P;
    fprintf('%4d    ', i);
    disp(q)
end
A = P'-eye(10);
A(1,:) = 1;
q = A\[1;zeros(9,1)];
fprintf('FINAL VECTOR');
disp(q')
Hr = 0;
for b=1:10
    Hr = Hr+q(b)*H(P(b,:));
end
disp(Hr)