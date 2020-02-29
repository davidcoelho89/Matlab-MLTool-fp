function [DATAout] = label_encode(DATAin,OPTION)

% --- Adjust label for the problem  ---
%
%   [DATAout] = label_encode(DATAin,OPTION)
%
%	Input:
%       DATAin.
%           output = misfit labels for each sample [Nc x N] or [1 x N]
%       OPTION.
%           lbl = label encode option
%               0) label in = label out
%               1) from sequencial (1, 2, 3...) to [-1 and +1] ([-1 -1 +1])
%               2) from sequencial (1, 2, 3...) to [0 and +1] ([0 0 +1])
%               3) from [-1 e +1] or [0 e 1] to sequencial (1, 2, 3...)
%               4) from string (words) to sequencial (1, 2, 3...)
%               5) from string (words) to -1 and +1 ([-1 -1 +1])
%               6) from string (words) to 0 and +1 ([0 0 +1])
%	Output:
%       DATAout.
%           output = adjusted labels [Nc x N] or [1 x N]

%% INITIALIZATIONS

option = OPTION.lbl;
labels_in = DATAin.output;   

%% ALGORITHM

switch(option)

case(0)

labels_out = labels_in;
    
case(1)

    [~,N] = size(labels_in);
    Nc = length(unique(labels_in));
    labels_out = -1*ones(Nc,N);
    for i = 1:N,
        labels_out(labels_in(i),i) = 1;
    end
    
case(2)
    
    [~,N] = size(labels_in);
    Nc = length(unique(labels_in));
    labels_out = zeros(Nc,N);
    for i = 1:N,
        labels_out(labels_in(i),i) = 1;
    end
    
case(3)
    
    [Nc,N] = size(labels_in);
    labels_out = zeros(1,N);
    for i = 1:N,
        for j = 1:Nc,
            if (labels_in(j,i) == 1)
                labels_out(i) = j;
            end
        end
    end
    
case(4)

    N = length(labels_in);
    
    Nc = length(unique(labels_in));
    class_names = unique(labels_in);
    
    labels_out = zeros(1,N);
    
    for i = 1:N,
        for j = 1:Nc,
            if strcmp(labels_in{i},class_names{j}),
                labels_out(i) = j;
                break;
            end
        end
    end
    
case(5)

    N = length(labels_in);
    
    Nc = length(unique(labels_in));
    class_names = unique(labels_in);
    
    labels_out = -1*ones(Nc,N);
    
    for i = 1:N,
        for j = 1:Nc,
            if strcmp(labels_in{i},class_names{j}),
                labels_out(j,i) = 1;
                break;
            end
        end
    end
    
case(6)

    N = length(labels_in);
    
    Nc = length(unique(labels_in));
    class_names = unique(labels_in);
    
    labels_out = zeros(Nc,N);
    
    for i = 1:N,
        for j = 1:Nc,
            if strcmp(labels_in{i},class_names{j}),
                labels_out(j,i) = 1;
                break;
            end
        end
    end
    
otherwise
	disp('Choose a correct option. Labels were not adjusted.')    
end

%% FILL OUTPUT STRUCTURE

DATAin.output = labels_out;
DATAout = DATAin;

%% END