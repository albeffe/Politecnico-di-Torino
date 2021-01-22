% Parametri iniziali
y = [1 1 1 1];
k = 2;
n = 4;
G = [1 0 1 1; 0 1 1 0];
p = 0.1;
v = [0 0; 0 1; 1 0; 1 1];

% Generazione matrice codewords
codebook = mod(v * G, 2);

% Calcolo probabilità
PR = [1; 1; 1; 1];
tot = 0;
for i = 1:size(codebook, 1)
    for j = 1:size(codebook, 2)
        if codebook(i, j) == y(j)
            D = 1 - p;
        else
            D = p;
        end
        PR(i) = PR(i) * D;
    end
    tot = tot + PR(i);
end

PR = PR/tot;

% Calcolo finale
probability_matrix = [0 0 0 0; 0 0 0 0];

for x = 1:2
    for j = 1:size(codebook, 2)
        for i = 1:size(codebook, 1)
            if codebook(i, j) == x - 1
               probability_matrix(x, j) = probability_matrix(x, j) + PR(i);
            end
        end
    end
end

probability_matrix



