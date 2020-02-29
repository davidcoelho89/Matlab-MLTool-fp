function [OUT] = bubble_sort(V,order)

% --- Bubble sorting algorithm ---
%
%   [OUT] = bubble_sort(in_vec,in_order)
%
%   Input:
%       V = vector to be organized
%       order = ascending (1) or descending order (2)
%   Output:
%       OUT.
%           V = organized vector
%           ind = holds indexes of original data

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(order))),
    order = 1;
end

%% INITIALIZATIONS

% Get vector size
len = length(V);

% Init Variables
ind = (1:len);

%% ALGORITHM

% Encreasing order
if order == 1,
    for i = 1:(len - 1),
        % Restore stop flag
        stop_flag = 1;
        % do bubbling
        for j = 1:(len - 1),
            if (V(j) > V(j+1)),
                % adjust input vector
                Vaux = V(j);
                V(j) = V(j+1);
                V(j+1) = Vaux;
                % adjust output order
                Vaux = ind(j);
                ind(j) = ind(j+1);
                ind(j+1) = Vaux;
                % clear stop flag
                stop_flag = 0;
            end
        end
        % breaks if stop flag is set 
        if stop_flag == 1,
            break;
        end
    end
% Decreasing order
else
    for i = 1:(len - 1),
        % Restore stop flag
        stop_flag = 1;
        % do bubbling
        for j = 1:(len - 1),
            if (V(j) < V(j+1)),
                % adjust input vector
                Vaux = V(j);
                V(j) = V(j+1);
                V(j+1) = Vaux;
                % adjust output order
                Vaux = ind(j);
                ind(j) = ind(j+1);
                ind(j+1) = Vaux;
                % clear stop flag
                stop_flag = 0;
            end
        end
        % breaks if stop flag is set 
        if stop_flag == 1,
            break;
        end
    end
end

%% FILL OUTPUT STRUCTURE

OUT.V = V;
OUT.ind = ind;

%% END