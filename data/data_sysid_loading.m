function [DATAout] = data_sysid_loading(OPTION)

% --- Selects a DataSet for System Identification Problems ---
%
%   [DATAout] = data_class_loading(OPTION)
%
%   Input:
%       OPTION.prob = which data base will be used
%           'linear_arx': Linear ARX problem
%       OPTION.input_type = specify data set
%           'prbs': type of input used (PseudoRandom Binary Signal)
%   Output:
%       DATA.
%           input = input signals       [Nu x N]
%           output = output signals     [Ny x N]

%% SET DEFAULT OPTIONS

if(nargin == 0 || (isempty(OPTION)))
    OPTION.prob = 'linear_arx';
    OPTION.input_type = 'prbs';
    OPTION.add_noise = 1;
    OPTION.noise_var = 0.01;
    OPTION.add_outlier = 0;
    OPTION.outlier_ratio = 0.05;
else
    if (~(isfield(OPTION,'prob')))
        OPTION.prob = 'linear_arx';
    end
    if (~(isfield(OPTION,'input_type')))
        OPTION.prob2 = 'prbs';
    end
    if (~(isfield(OPTION,'add_noise')))
        OPTION.add_noise = 1;
    end
    if (~(isfield(OPTION,'noise_var')))
        OPTION.noise_var = 0.01;
    end
    if (~(isfield(OPTION,'add_outlier')))
        OPTION.add_outlier = 0;
    end
    if (~(isfield(OPTION,'outlier_ratio')))
        OPTION.outlier_ratio = 0.05;
    end
    
end

%% INITIALIZATIONS

DATA = struct('input',[],'output',[]);

%% ALGORITHM

problem = OPTION.prob;

if (strcmp(problem,'linear_arx'))
    DATA.input = OPTION.input_ts;
    DATA.output = arxOutputFromInput(OPTION.input_ts,...
                                     OPTION.y_coefs, ...
                                     OPTION.u_coefs, ...
                                     OPTION.noise_var);

end

%% FILL OUTPUT STRUCTURE

DATAout = DATA;

%% END






























