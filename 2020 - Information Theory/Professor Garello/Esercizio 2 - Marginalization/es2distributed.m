clc
clear all
close all

% Parametri iniziali
k = 2;
n = 4;
v = [0 0; 0 1; 1 0; 1 1];
y = [1 1 0 1];
p = 0.1;
R = [1 1; 1 0];
G = [eye(2) R];
H = [R eye(2)]';
iter = 2;
distributed_probability_matrix = [0 0 0 0; 0 0 0 0];

% Brute Force ----------------------------------------------------------
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
brute_force_probability_matrix = [0 0 0 0; 0 0 0 0];

for x = 1:2
    for j = 1:size(codebook, 2)
        for i = 1:size(codebook, 1)
            if codebook(i, j) == x - 1
               brute_force_probability_matrix(x, j) = brute_force_probability_matrix(x, j) + PR(i);
            end
        end
    end
end

% End Brute Force ------------------------------------------------------

% Distributed ----------------------------------------------------------

% Inizializzazione matrici messaggi C to F di 0 e 1 bits
C_F_0 = ([y' y'] == [1 1])*p.*H + ([y' y'] == [0 0])*(1-p).*H;
C_F_1 = ([y' y'] == [1 1])*(1-p).*H + ([y' y'] == [0 0])*p.*H;

% Inizializzazione matrice messaggi F to C di 0 bit
F_C_0 = zeros(size(C_F_0));

% Iterazione al fine di raggiungere convergenza (in assenza di loops)
for iterazioni = 0:iter
    
    % Creazione matrice messaggi F to C di bit 0 attraverso la formula
    % "Trick"
    for i=1:size(C_F_0,1)
        factors = C_F_1(1:end~=i, :);
        product = prod(1-2.*factors,1);
        msg = (1 + product)./2;
        F_C_0(i,:) = msg;
    end

    % Rimozione degli zeri e creazione matrice messaggi F to C di bit 1
    F_C_0 = F_C_0.*H;
    F_C_1 = ones(size(F_C_0));
    F_C_1 = (1 - F_C_0).*H;        
    
    % Calcolo prodottoria di tutti i messaggi entranti nel nodo
    prdct0 = [1 1 1 1];
    prdct1 = [1 1 1 1];

    for i=1:size(F_C_0,1)
    
        if y(i) == 1
            prdct0(i) = prdct0(i) * 0.1;
            prdct1(i) = prdct1(i) * 0.9;
        else
            prdct0(i) = prdct0(i) * 0.9;
            prdct1(i) = prdct1(i) * 0.1;
        end
    
        for j=1:size(F_C_0,2)
        
            if F_C_0(i, j) ~= 0
                prdct0(i) = prdct0(i) * F_C_0(i, j);
                prdct1(i) = prdct1(i) * F_C_1(i, j);
            end
        
        end
    
    end
    
    % Calcolo risultato
    distributed_probability_matrix = [prdct0; prdct1];
    % Normalizzazione risultato
    for j=1:size(distributed_probability_matrix,2)
            distributed_probability_matrix(1, j) = distributed_probability_matrix(1, j) / (distributed_probability_matrix(1, j) + distributed_probability_matrix(2, j));
            distributed_probability_matrix(2, j) = 1 - distributed_probability_matrix(1, j);
    end

    % Calcolo matrici C to F mediante extrisic info + Normalizzazione
    for i=1:size(F_C_0,1)
    
        for j=1:size(F_C_0,2)
            if F_C_0(i, j) ~= 0
                C_F_0(i, j) = prdct0(i) / F_C_0(i, j);
                C_F_1(i, j) = prdct1(i) / F_C_1(i, j);
                C_F_0(i, j) = C_F_0(i, j) / (C_F_0(i, j) + C_F_1(i, j));
                C_F_1(i, j) = C_F_1(i, j) / (C_F_0(i, j) + C_F_1(i, j));
            end
        end
    end
    C_F_1 = (1 - C_F_0).*H;

end

% End Distributed ------------------------------------------------------

% Show Results ---------------------------------------------------------
brute_force_probability_matrix
distributed_probability_matrix