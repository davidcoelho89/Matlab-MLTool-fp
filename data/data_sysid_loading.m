function [DATAout] = data_sysid_loading(OPTION)

% --- Selects a DataSet for System Identification Problems ---
%
%   [DATAout] = data_class_loading(OPTION)
%
%   Input:
%       OPTION.prob = which data set will be used
%           'linear_arx': Linear ARX problem
%           'tank': two cascaded-tanks
%           'actuator': 
%           'exchanger': 
%       OPTION.prob2 = specify data set
%           databases that use this field:
%           'linear_arx', ...
%   Output:
%       DATA.
%           input = input signals       [Nu x N]
%           output = output signals     [Ny x N]

%% SET DEFAULT OPTIONS

if(nargin == 0 || (isempty(OPTION)))
    OPTION.prob = 'linear_arx';
    OPTION.prob2 = 01;
else
    if (~(isfield(OPTION,'prob')))
        OPTION.prob = 'linear_arx';
    end
    if (~(isfield(OPTION,'prob2')))
        OPTION.prob2 = 01;
    end
end

%% INITIALIZATIONS

DATA = struct('input',[],'output',[]);

%% ALGORITHM

problem = OPTION.prob;
problem_spec = OPTION.prob2;

if (strcmp(problem,'linear_arx'))
    if (problem_spec == 01) % y[n] = 0.4y[n-1] - 0.6y[n-2] + u[n-1]
        loaded_data = load('linear_arx_01.mat');
        DATA.input = loaded_data.u_ts';
        DATA.output = loaded_data.y_ts';
    end
elseif(strcmp(problem,'tank'))
    loaded_data = load('Tank2.mat');
    DATA.input = loaded_data.u;
    DATA.output = loaded_data.y;
elseif(strcmp(problem,'actuator'))
    loaded_data = load('actuator.dat');
    DATA.input = loaded_data(:,1)';
    DATA.output = loaded_data(:,2)';
elseif(strcmp(problem,'exchanger'))
    loaded_data = load('exchanger.dat');
    DATA.input = loaded_data(:,1)';
    DATA.output = loaded_data(:,2)';
else
    disp('Choose an existing data set. Empty signals were created.');
end

%% FILL OUTPUT STRUCTURE

DATAout = DATA;

%% END






























