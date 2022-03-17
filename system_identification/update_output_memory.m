function updated_memory = update_output_memory(Y,memory,lags)

%
%
%

%% INITIALIZATIONS

Ny = length(lags);                       % Number of outputs
memory_length = length(memory);     	 % Sum of lags from all outputs 
updated_memory = zeros(memory_length,1); % Init update memory

initial_sample = 1;     % Initial_sample for first output memory

%% ALGORITHM

for i = 1:Ny
    % Get lag and last sample for present output
    lag = lags(i);
    final_sample = initial_sample + lag - 1;
    % Update memory from specific output
    if(lag == 1)
        updated_memory(initial_sample) = Y(i);
    else
        updated_memory(initial_sample:final_sample) = ...
        [Y(i),memory(initial_sample:final_sample-1)];
    end
    % Update Initial sample for next output
    initial_sample = final_sample + 1;
end


%% END