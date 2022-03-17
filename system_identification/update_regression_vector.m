function x_out = update_regression_vector(x,out_mem,pred_type,add_bias)

%
%
%

%% INITIALIZATIONS

outputs_mem_length = length(out_mem);

x_out = x;

%% ALGORITHM

if(pred_type == 0)       % free simulation
    if(add_bias)
        x_out(2:outputs_mem_length+1) = out_mem;
    else
        x_out(1:outputs_mem_length) = output_mermory;
    end    
elseif (pred_type == 1) % 1-step ahead
    % Does nothing
else                    % n-steps ahead
    % ToDo - Verify
end


%% END