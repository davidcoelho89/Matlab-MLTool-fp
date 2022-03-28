function xn_out = update_regression_vector(xn,out_mem,pred_type,add_bias)

%
%
%

%% INITIALIZATIONS

outputs_mem_length = length(out_mem);

xn_out = xn;

%% ALGORITHM

if(pred_type == 0)       % free simulation
    if(add_bias)
        xn_out(2:outputs_mem_length+1,1) = out_mem;
    else
        xn_out(1:outputs_mem_length,1) = out_mem;
    end    
elseif (pred_type == 1) % 1-step ahead
    % Does nothing
else                    % n-steps ahead
    % ToDo - Verify
end


%% END