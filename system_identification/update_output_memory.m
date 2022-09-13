function updated_memory = update_output_memory(Y,outdated_memory,lags)

%
%
%

%% INITIALIZATIONS

Ny = length(lags);                       % Number of outputs
memory_length = length(outdated_memory); % Sum of lags from all outputs 
updated_memory = zeros(memory_length,1); % Init update memory

initial_sample_index = 1;

%% ALGORITHM

for i = 1:Ny
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