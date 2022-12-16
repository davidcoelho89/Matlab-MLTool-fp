function updated_memory = update_output_memory(Y,outdated_memory,lags)

%% INITIALIZATIONS

% Init update memory
updated_memory = zeros(length(outdated_memory),1); 

%% ALGORITHM

initial_sample_index = 1;
for i = 1:length(lags)
    % Get lag and last sample for present output
    lag = lags(i);
    final_sample_index = initial_sample_index + lag - 1;
    % Update memory from specific output
    if(lag == 1)
        updated_memory(initial_sample_index) = Y(i);
    else
        updated_memory(initial_sample_index:final_sample_index,1) = ...
        [Y(i); outdated_memory(initial_sample_index:final_sample_index-1,1)];
    end
    % Update Initial sample for next output
    initial_sample_index = final_sample_index + 1;
end

%% END