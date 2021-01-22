function Vettore_decodificato = viterbi(vettore_da_decodificare, D)
    Vettore_decodificato = [];
    
    vettore_traiettoria_1 = [];
    vettore_traiettoria_2 = [];
    
    pesi_stati = zeros(2, 1);
    indice = 1;
    
        for i=1:2:size(vettore_da_decodificare, 2)
        y = vettore_da_decodificare(i:i+1);
       
        matrice_pesi = [sum(y ~= [0 0]) sum(y ~= [0 1]); sum(y ~= [1 1]) sum(y ~= [1 0])];
       
        if i==1
            matrice_pesi(2,1) = 5;
            matrice_pesi(2,2) = 5;
        end
       
        [nuovo_peso_1, indice_minimo_1] = min([pesi_stati(1,1) + matrice_pesi(1,1) pesi_stati(2,1) + matrice_pesi(2,1)]);
        [nuovo_peso_2, indice_minimo_2] = min([pesi_stati(1,1) + matrice_pesi(1,2) pesi_stati(2,1) + matrice_pesi(2,2)]);
       
        varia = vettore_traiettoria_1;
        if indice_minimo_1==1
            vettore_traiettoria_1 = [vettore_traiettoria_1 0];
        else
            vettore_traiettoria_1 = [vettore_traiettoria_2 0];
        end
       
        if indice_minimo_2==1
            vettore_traiettoria_2 = [varia 1];
        else
            vettore_traiettoria_2 = [vettore_traiettoria_2 1];
        end
       
        pesi_stati = [nuovo_peso_1; nuovo_peso_2];
           
        if nuovo_peso_1 < nuovo_peso_2
            vet = vettore_traiettoria_1;
        else
            vet = vettore_traiettoria_2;
        end
    
        if i>2 * D
            Vettore_decodificato = [Vettore_decodificato vet(indice)];
            indice = indice + 1;
        end
    end
end