function [dataset_out] = encodeLabels(dataset_in,label_encoding)

% --- Adjust labels' codes  ---
%
%   [dataset_out] = encodeLabels(dataset_in,label_encoding)
%
%	Input:
%       dataset_in.
%           output = misfit labels for each sample [Nc x N] or [1 x N]
%       label_encoding
%           = 'bipolar' = [-1 and +1]. Ex: ([-1 -1 +1]
%           = 'binary' = [0 and +1]. Ex: ([0 0 +1])
%           = 'sequential' = (1, 2, 3...)
%	Output:
%       DATAout.
%           output = adjusted labels [Nc x N] or [1 x N]

%% ALGORITHM

labels_in = dataset_in.output;   

% ToDo - Verify if it is numerical or literal 

if(strcmp(label_encoding,'original'))
    labels_out = labels_in;
elseif(strcmp(label_encoding,'bipolar'))
	[~,N] = size(labels_in);
    Nc = length(unique(labels_in));
    labels_out = -1*ones(Nc,N);
    for i = 1:N
        labels_out(labels_in(i),i) = 1;
    end
end

%% FILL OUTPUT STRUCTURE

dataset_out = dataset_in;
dataset_out.output = labels_out;

%% ALGORITHM

% 0) label in = label out
% 1) from sequencial  to )
% 2) from sequencial (1, 2, 3...) to 
% 3) from [-1 e +1] or [0 e 1] to sequencial (1, 2, 3...)
% 4) from string (words) to sequencial (1, 2, 3...)
% 5) from string (words) to -1 and +1 ([-1 -1 +1])
% 6) from string (words) to 0 and +1 ([0 0 +1])

% switch(label_encoding)
% 
% case(1)
% 
%     [~,N] = size(labels_in);
%     Nc = length(unique(labels_in));
%     labels_out = -1*ones(Nc,N);
%     for i = 1:N
%         labels_out(labels_in(i),i) = 1;
%     end
%     
% case(2)
%     
%     [~,N] = size(labels_in);
%     Nc = length(unique(labels_in));
%     labels_out = zeros(Nc,N);
%     for i = 1:N
%         labels_out(labels_in(i),i) = 1;
%     end
%     
% case(3)
%     
%     [Nc,N] = size(labels_in);
%     labels_out = zeros(1,N);
%     for i = 1:N
%         for j = 1:Nc
%             if (labels_in(j,i) == 1)
%                 labels_out(i) = j;
%             end
%         end
%     end
%     
% case(4)
% 
%     N = length(labels_in);
%     
%     Nc = length(unique(labels_in));
%     class_names = unique(labels_in);
%     
%     labels_out = zeros(1,N);
%     
%     for i = 1:N
%         for j = 1:Nc
%             if strcmp(labels_in{i},class_names{j})
%                 labels_out(i) = j;
%                 break;
%             end
%         end
%     end
%     
% case(5)
% 
%     N = length(labels_in);
%     
%     Nc = length(unique(labels_in));
%     class_names = unique(labels_in);
%     
%     labels_out = -1*ones(Nc,N);
%     
%     for i = 1:N
%         for j = 1:Nc
%             if strcmp(labels_in{i},class_names{j})
%                 labels_out(j,i) = 1;
%                 break;
%             end
%         end
%     end
%     
% case(6)
% 
%     N = length(labels_in);
%     
%     Nc = length(unique(labels_in));
%     class_names = unique(labels_in);
%     
%     labels_out = zeros(Nc,N);
%     
%     for i = 1:N
%         for j = 1:Nc
%             if strcmp(labels_in{i},class_names{j})
%                 labels_out(j,i) = 1;
%                 break;
%             end
%         end
%     end
%     
% otherwise
% 	disp('Choose a correct option. Labels were not adjusted.')    
% end
% 
% %% FILL OUTPUT STRUCTURE
% 
% dataset_in.output = labels_out;
% dataset_out = dataset_in;

%% END