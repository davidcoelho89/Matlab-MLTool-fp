function [STATS_acc_out] = calculate_binary_stats(STATS_acc_in,class_1_vect)

%% INIT

number_of_repetitions = length(STATS_acc_in);

STATS_acc_out = cell(number_of_repetitions,1);

[number_of_classes,~] = size(STATS_acc_in{1}.Mconf);

%% ALGORITHM

for n = 1:number_of_repetitions
    
    % Init Binary Confusion Matrix
    STATS_acc_out{n}.Mconf = zeros(2,2);

    % Calculate Binary Confusion Matrix
    Mconf_in = STATS_acc_in{n}.Mconf;
    for i = 1:number_of_classes
        for j = 1:number_of_classes
            if any(class_1_vect == i)
                lin = 1;
            else
                lin = 2;
            end
            if any(class_1_vect == j)
                col = 1;
            else
                col = 2;
            end

            STATS_acc_out{n}.Mconf(lin,col) = ...
                 STATS_acc_out{n}.Mconf(lin,col) + Mconf_in(i,j);
        end
    end

    % Calculate Accuracy and Error
    number_of_samples = sum(sum(STATS_acc_out{n}.Mconf));
    STATS_acc_out{n}.acc = (STATS_acc_out{n}.Mconf(1,1) + ...
                            STATS_acc_out{n}.Mconf(2,2)) / ...
                            number_of_samples;
    STATS_acc_out{n}.err = 1 - STATS_acc_out{n}.acc;

end

end