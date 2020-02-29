function decimal = bin2deci(binario)

% --- Convert Binary Vector to Decimal ---
%
% 	decimal = bin2deci(binario)
%
%   most significant bit: left
%   least significant bit: right

%% INITIALIZATIONS

n_bits = length(binario);

%% ALGORITHM

decimal = 0;
for bit = 1:n_bits,
   decimal = decimal + binario(bit)*(2^(n_bits-bit));
end

%% END