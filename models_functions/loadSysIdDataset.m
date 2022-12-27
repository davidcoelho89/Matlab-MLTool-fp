function [dataset] = loadSysIdDataset(dataset_name)

% --- Selects a DataSet for System Identification Problems ---
%
%   [dataset] = loadSysIdDataset(dataset_name)
%
%   Input:
%       dataset_name = which data set will be used
%           'linear_arx_01': Linear ARX problem
%           'tank': two cascaded-tanks
%           'actuator': 
%           'exchanger': 
%           'motor_step': 
%           'motor': 
%   Output:
%       dataset.
%           input = input signals       [Nu x N]
%           output = output signals     [Ny x N]

%% SET DEFAULT OPTIONS

if(nargin == 0)
    dataset_name = 'linear_arx_01';
end

%% INITIALIZATIONS

DATA = struct('input',[],'output',[],'name',dataset_name);

%% ALGORITHM

if (strcmp(dataset_name,'linear_arx_01'))
    % y[n] = 0.4y[n-1] - 0.6y[n-2] + 2.0u[n-1]
    loaded_data = load('linear_arx_01.mat');
    DATA.input = loaded_data.u_ts';
    DATA.output = loaded_data.y_ts';
elseif(strcmp(dataset_name,'tank'))
    loaded_data = load('Tank2.mat');
    DATA.input = loaded_data.u;
    DATA.output = loaded_data.y;
elseif(strcmp(dataset_name,'actuator'))
    loaded_data = load('actuator.dat');
    DATA.input = loaded_data(:,1)';
    DATA.output = loaded_data(:,2)';
elseif(strcmp(dataset_name,'exchanger'))
    loaded_data = load('exchanger.dat');
    DATA.input = loaded_data(:,1)';
    DATA.output = loaded_data(:,2)';
elseif(strcmp(dataset_name,'motor_step'))
	loaded_data = load('motor2.mat');
    DATA.input = loaded_data.u;
    DATA.output = loaded_data.y;
elseif(strcmp(dataset_name,'motor_aprbs'))
	loaded_data = load('motor_aprbs.mat');
    DATA.input = loaded_data.u1;
    DATA.output = loaded_data.y1;
elseif(strcmp(dataset_name,'motor'))
	loaded_data = load('motor.mat');
    DATA.input = loaded_data.u1;
    DATA.output = loaded_data.y1;
else
    disp('Choose an existing data set. Empty signals were created.');
end

%% FILL OUTPUT STRUCTURE

dataset = DATA;

%% END