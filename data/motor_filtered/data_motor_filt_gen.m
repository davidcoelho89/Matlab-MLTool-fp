function [DATAout] = data_motor_filt_gen(OPTION)

% --- Generates data for motor problem  ---
%
%   [DATAout] = data_motor_gen(OPTION)
%
%   Input:
%      OPTION.prob2 = Problem definition
%   Output:
%       DATAout = general data
%           .input   = attributes' matrix [pxN]
%           .output  = labels' matrix [1xN]
%                      (with just 1 value - 1 to Nc)
%                      (includes ordinal classification) 
%           .lbl     = labels' vector [1xN]
%                      (original labels of data set)

%% INITIALIZATIONS

problem = OPTION.prob2;

input = [];
output = [];
lbl = [];

%% ALGORITHM

switch(problem)
    
    case 1,
        % Load filtered Motor Data 1
        load data_filt_01_red.mat;
        input = data01_red.data_d;
        output = data01_red.labels(1,:) + 1;
        lbl = data01_red.labels(1,:) + 1;
        clear data01_red;
    case 2,
        % Load filtered Motor Data 2
        load data_filt_02_red.mat;
        input = data02_red.data_d;
        output = data02_red.labels(1,:) + 1;
        lbl = data02_red.labels(1,:) + 1;
        clear data02_red;
    case 3,
        % Load filtered Motor Data 3
        load data_filt_03_red.mat;
        input = data03_red.data_d;
        output = data03_red.labels(1,:) + 1;
        lbl = data03_red.labels(1,:) + 1;
        clear data03;
    otherwise,
        disp('Choose a correct option. Void data was generated.')
end

%% FILL OUTPUT STRUCTURE

DATAout.input = input;
DATAout.output = output;
DATAout.lbl = lbl;

%% END