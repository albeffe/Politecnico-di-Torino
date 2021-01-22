function vettore_codificato = codifica_trellis(vettore_da_codificare)

    trellis_arco = [0 1; 3 2]; % 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
    trellis_architettura = [1 2; 1 2]; % 00 e 11 fanno andare a stato alto
    
    stato_corrente = 1;
    vettore_codificato = [];

    for i=1:size(vettore_da_codificare, 2)
        codifica = flip(de2bi(trellis_arco(stato_corrente, vettore_da_codificare(i) + 1), 2));
        vettore_codificato = [vettore_codificato codifica];
        stato_corrente = trellis_architettura(stato_corrente, vettore_da_codificare(i) + 1);
    end

end