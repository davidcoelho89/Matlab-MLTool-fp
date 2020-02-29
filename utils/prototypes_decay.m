function [out_decay] = prototypes_decay(decay,Nn,neig,t,epoch)

% --- Neigborhood decreasing function ---
% 
%   [out_decay] = som_f_decay(decay,Nn,neig,t)
% 
%   Input:
%       decay = use or not this function
%           1: use it
%       Nn = number of neighbors [1x1]
%       neig = type of neighborhood function [1x1]
%       t = time [1x1]
%   Output:
%       out_decay.
%           Nn = number of neighbors [1x1]
%           neig = type of neighborhood function [1x1]
%           t = time [1x1]

%% INIT

% Don't Need

%% ALGORITHM

% In case decay function is on
if decay == 1,
    if epoch <= 2,
        out_decay.Nn = 4;
     	out_decay.neig = 1;
     	out_decay.t = 0;
   	elseif epoch <= 4,
       	out_decay.Nn = 3;
       	out_decay.neig = 1;
       	out_decay.t = 0;
   	elseif epoch <= 6,
       	out_decay.Nn = 2;
       	out_decay.neig = 1;
     	out_decay.t = 0;
  	elseif epoch <= 8,
       	out_decay.Nn = 1;
      	out_decay.neig = 1;
       	out_decay.t = 0;
    else
      	out_decay.Nn = 1;
     	out_decay.neig = 2;
        out_decay.t = t;
    end
% In other cases,    
else
 	out_decay.Nn = Nn;
  	out_decay.neig = neig;
  	out_decay.t = t;
end


%% FILL OUTPUT STRUCTURE

% Don't Need

%% END