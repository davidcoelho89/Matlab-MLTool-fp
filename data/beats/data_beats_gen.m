function [DATAout] = data_beats_gen(OPTION)

% --- Generates data for Beats Dataset  ---
%
%   [DATAout] = data_beats_gen(OPTION)
%
%   Input:
%      OPTION.prob2 = Problem definition
%   Output:
%       DATAout = general data
%           .input   = attributes' matrix                   [pxN]
%           .output  = labels' matrix                       [1xN]
%                      (with just 1 value - 1 to Nc)
%                      (includes ordinal classification) 
%           .lbl     = labels' vector                       [1xN]
%                      (original labels of data set)

%% SET DEFAULT OPTIONS

if (~(isfield(OPTION,'prob2')))
    OPTION.prob2 = 0;
end

%% INITIALIZATIONS

problem = OPTION.prob2;

DATA = struct('input',[],'output',[],'lbl',[]);

% disp('passei 1');
% disp(problem);

%% ALGORITHM

if (strcmp(problem,'original'))
    loaded_data = load('Beats_Dataset.mat');
    DATA.input = loaded_data.Beats_Dataset(:,1:42)';
    DATA.output = loaded_data.Beats_Dataset(:,43)';
    DATA.lbl = DATA.output;
else
    loaded_data = load(problem);
    
    if ((isfield(loaded_data,'Beat_Dataset_Train')))
        DATA.input = loaded_data.Beat_Dataset_Train(:,1:42)';
        DATA.output = loaded_data.Beat_Dataset_Train(:,43)';
    
    elseif ((isfield(loaded_data,'Beat_Dataset_Test')))
        DATA.input = loaded_data.Beat_Dataset_Test(:,1:42)';
        DATA.output = loaded_data.Beat_Dataset_Test(:,43)';
    
    elseif ((isfield(loaded_data,'dataset_train')))
        DATA.input = loaded_data.dataset_train{:,1:42};
        DATA.input = DATA.input';
        [~,N] = size(DATA.input);
        DATA.output = zeros(1,N);
        dataset_output = loaded_data.dataset_train{:,43};
        for i = 1:N
            if(strcmp(dataset_output{i},'class0'))
                DATA.output(i) = 1;
            else
                DATA.output(i) = 2;
            end
        end
        
    elseif ((isfield(loaded_data,'dataset_test')))
        DATA.input = loaded_data.dataset_test{:,1:42};
        DATA.input = DATA.input';
        [~,N] = size(DATA.input);
        DATA.output = zeros(1,N);
        dataset_output = loaded_data.dataset_test{:,43};
        for i = 1:N
            if(strcmp(dataset_output{i},'class0'))
                DATA.output(i) = 1;
            else
                DATA.output(i) = 2;
            end
        end
        
    end
    
    DATA.lbl = DATA.output;
end

DATA.name = 'beats';

%% FILL OUTPUT STRUCTURE

DATAout = DATA;

%% END


























