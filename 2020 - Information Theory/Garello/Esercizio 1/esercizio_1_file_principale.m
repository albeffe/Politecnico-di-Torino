clc

% Parametri:
lunghezza = 100; % lunghezza della sequenza di bit random
valori_delay = [lunghezza 2 5]; % valori di delay che assumerà il programma
% delay = lunghezza rappresenta il viterbi senza early decision

for i=1:3
    D = valori_delay(i);
    ber = +inf;
    vettore_ber = [];
    snr_db = 0;
    
    while snr_db <= 4
        bit_generati = 0;
        bit_sbagliati = 0;
    
        while bit_sbagliati < 100
        
            % Vettore Random
            r = [randi([0 1], 1, lunghezza) zeros(1, D)];
            
            % Codifica Trellis
            encoded_vec = codifica_trellis(r);

            % Modulazione
            a = 1;
            mododulated_vec = ones(1, size(encoded_vec, 2) )* a;
            for i=1:size(encoded_vec, 2)
                if encoded_vec(i) == 0
                    mododulated_vec(i) = -a;
                end
            end

            % Aggiunta Rumore
            snr = 10^(snr_db / 10);
            s = 1/sqrt(2 * snr);
            rumore = randn(1, size(mododulated_vec, 2)) * s;
            modulated_noized_vec = mododulated_vec + rumore;

            % PSK Decision
            a = 1;
            post_psk_vec = ones(1, size(modulated_noized_vec, 2));
            for i=1:size(modulated_noized_vec, 2)
                if modulated_noized_vec(i)<0
                    post_psk_vec(i) = 0;
                end
            end
        
            % Decodifica Viterbi
            viterbi_decodifica = viterbi(post_psk_vec, D);
        
            % Aggiorno i contatori
            bit_generati = bit_generati + size(r(1:end - D), 2);
            bit_sbagliati = bit_sbagliati + sum(r(1:end - D) ~= viterbi_decodifica);        
        end
    
        % Calcolo ber
        ber = bit_sbagliati / bit_generati;
        vettore_ber = [vettore_ber ber];
    
        snr_db = snr_db + 1;
        disp('Calcolo in corso')
    end
    semilogy(vettore_ber)
    hold on
end
hold off
legend('Normal Viterbi', 'Delay = 2', 'Delay = 5')
