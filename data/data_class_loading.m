    function [DATAout] = data_class_loading(OPTION)

% --- Selects a Data Base ---
%
%   [DATAout] = data_class_loading(OPTION)
%
%   Input:
%       OPTION.prob = which data base will be used
%           01: boxes
%           02: dermatology
%           03: Yale A faces
%           04: four groups
%           05: images data
%           06: iris data
%           07: motor short circuit failure
%           08: motor short circuit filtered
%           09: random
%           10: Vertebral Column 
%           11: Two Moons
%           12: Wine
%           13: Motor broken bar failure multiclass
%           14: Motor broken bar failure binary class
%           15: Breast Cancer
%           16: Cryotherapy
%           17: Immunotherapy
%           18: Abalone
%           19: Cervical Cancer
%           20: Sensorless Drive
%           21: Mnist Digits
%           22: Wall-following Robot
%           23: EEG Eye State
%           24: Werable Movements
%           25: SEA Concepts
%           26: Rotating Hyperplane
%           27: RBF moving
%           28: RBF interchanging
%           29: Moving Squares
%           30: Chess Transient
%           31: Mixed Drift
%           32: Led
%           33: Weather
%           34: Electricity Market
%           35: Cover Type
%           36: Poker
%           37: Outdoor
%           38: Rialto
%           39: Spam
%       OPTION.prob2 = specify data set
%           databases that use this field: 03, 07, 08, 10, 19 
%   Output:
%       DATA.
%           input = input matrix            [p x N]
%           output = output matrix          [1 x N] (sequential: 1, 2...)
%           lbl = mantain original labels   (usually sequential)

%% INITIALIZATION

DATA = struct('input',[],'output',[],'lbl',[]);

choice = OPTION.prob;

%% ALGORITHM

switch (choice),
    
    case 1,
        % Load Boxes Data
        loaded_data1 = load('data_boxes_train.dat');
        loaded_data2 = load('data_boxes_class.dat');
        DATA.input = [loaded_data1(:,1:end-1)' loaded_data2(:,1:end-1)'];  
        DATA.output = [loaded_data1(:,end)' loaded_data2(:,end)'];
        DATA.lbl = DATA.output;                  % Original Labels
    case 2,
        % Load Dermatology Data
        loaded_data = importdata('data_dermato_03.txt');
        DATA.input = loaded_data(:,1:end-1)';	% Input
        DATA.output = loaded_data(:,end)';    	% Output
        DATA.lbl = DATA.output;               	% Original Labels
    case 3,
        % Load Faces Data (YALE A)
        DATA = face_preprocess_col(OPTION);
    case 4,
        % Load Four Groups Data
        load data_four_groups.mat;
        DATAaux.input = DATA.dados;
        DATAaux.output = DATA.alvos;
        DATAaux.lbl = DATA.rot;
        DATA = DATAaux;
    case 5,
        % Load Images Data
        disp('Still Not implemented. Void Structure Created')
    case 6,
        % Load Iris Data
        loaded_data = importdata('data_iris.m');
        DATA.input = loaded_data(:,1:end-1)';
        DATA.output = loaded_data(:,end)';
        DATA.lbl = DATA.output;
    case 7,
        % Load Motor Failure Data
        if (isfield(OPTION,'prob2'))
            DATA = data_motor_gen(OPTION);
        else
            disp('Specify the database')
        end
    case 8,
        % Load Motor Failure filtered Data
        if(isfield(OPTION,'prob2')),
            DATA = data_motor_filt_gen(OPTION);
        else
            disp('Specify the database')
        end
    case 9,
        % Load Random Data
        DATA.input = rand(2,294);
        DATA.output = ceil(3*rand(1,294));
        DATA.lbl = DATA.output;
    case 10,
        % Load Vertebral Column Data
        spine = importdata('data_spine.mat');
        DATA.input = spine(:,1:6)';
        DATA.output = spine(:,7:9)';
        DATA.lbl = DATA.output;
        OPT.lbl = 3;
        DATA = label_adjust(DATA,OPT);
        if OPTION.prob2 == 2,
            DATA.output(1:210) = 2;
            DATA.output(211:end) = 1;
            DATA.lbl = DATA.output;
        end
    case 11,
        % Load Two Moons Data
        loaded_data = load('data_two_moons.dat');
        DATA.input = loaded_data(:,1:2)';
        loaded_data(502:end,3) = 2;
        DATA.output = loaded_data(:,3)';
        DATA.lbl = DATA.output;
    case 12,
        % Load Wine Data
        loaded_data = importdata('data_wine_03.m');
        DATA.input = loaded_data(:,1:end-1)';	% Input
        DATA.output = loaded_data(:,end)';   	% Output
        DATA.lbl = DATA.output;               	% Rotulos
    case 13,
        % Load Motor broken bar multi class data
        DATA_aux = load('data_bb.mat');
        DATA.input = DATA_aux.DATA.input;
        DATA.output = DATA_aux.DATA.output;
        DATA.lbl = DATA.output;
    case 14,
        % Load Motor broken bar binary class data
        DATA_aux = load('data_bb.mat');
        DATA.input = DATA_aux.DATA.input;
        output = DATA_aux.DATA.output;
        DATA.lbl = output;
        for i = 1:2520,
           if output(i) ~= 1,
               output(i) = 2;
           end
        end
        DATA.output = output;
    case 15,
        loaded_data = importdata('data_breast_cancer.txt');
        data_aux = zeros(683,11);
        cont = 0;
        for i = 1:699,
            if (loaded_data(i,8) ~= 0),
                cont = cont + 1;
                data_aux(cont,:) = loaded_data(i,:);
            end
        end
        DATA.input = data_aux(:,3:11)';
        DATA.output = data_aux(:,2)'/2;
        DATA.lbl = DATA.output;
    case 16,
        loaded_data = importdata('data_cryotherapy.txt');
        DATA.input = loaded_data(:,1:6)';
        DATA.output = loaded_data(:,7)'+1;
       	DATA.lbl = DATA.output;
    case 17,
        loaded_data = importdata('data_immunotherapy.txt');
        DATA.input = loaded_data(:,1:7)';
        DATA.output = loaded_data(:,8)'+1;
       	DATA.lbl = DATA.output;
    case 18,
        loaded_data = importdata('abalone.mat');
        DATA.input = loaded_data(:,1:8)';
        loaded_data(481,9) = 28;
        DATA.output = loaded_data(:,9)';
        DATA.lbl = DATA.output;
    case 19,
        loaded_data = importdata('cervical_cancer.mat');
        DATA.input = loaded_data.input;
        DATA.output = loaded_data.output;
        DATA.lbl = DATA.output;
        if (OPTION.prob2 == 2),
            DATA.output(1:242) = 1;
            DATA.output(243:end) = 2;
            DATA.lbl = DATA.output;
        end
    case 20,
        loaded_data = importdata('sensorless_drive_diagnosis.txt');
        DATA.input = loaded_data(:,1:48)';
        DATA.output = loaded_data(:,49)';
        DATA.lbl = DATA.output;
    case 21,
        load('mnist.mat');
        DATA.input = mnist_data;
        DATA.output = mnist_lbl + 1;
        DATA.lbl = DATA.output;
    case 22,
        if(isfield(OPTION,'prob2')),
            db = OPTION.prob2;
        else
            db = 1;
        end
        switch db
            case 1
                sensors = load('sensor_readings_2.data');
            case 2
                sensors = load('sensor_readings_4.data');
            case 3
                sensors = load('sensor_readings_24.data');
            otherwise
                sensors = load('sensor_readings_2.data');
        end
     	DATA.input = sensors(:,1:end-1)';
      	DATA.output = sensors(:,end)';
      	DATA.lbl = sensors(:,end)';
    case 23,
        loaded_data = importdata('EEGeyeState.mat');
        DATA.input = loaded_data.input;
        DATA.output = loaded_data.output + 1;
        DATA.lbl = DATA.output;
    case 24,
        loaded_data = importdata('werable.mat');
        DATA.input = loaded_data.input;
        DATA.output = loaded_data.output;
        DATA.lbl = DATA.output;
    case 25,
        loaded_data = load('sea.mat');
        DATA.input = loaded_data.SEAdata';
        DATA.output = loaded_data.SEAclass' + 1;
        DATA.lbl = DATA.output;
    case 26,
        DATA.input = load('rotatingHyperplane.data')';
        DATA.output = load('rotatingHyperplane.labels')' + 1;
        DATA.lbl = DATA.output;
    case 27,
        DATA.input = load('movingRBF.data')';
        DATA.output = load('movingRBF.labels')' + 1;
        DATA.lbl = DATA.output;
    case 28,
        DATA.input = load('interchangingRBF.data')';
        DATA.output = load('interchangingRBF.labels')' + 1;
        DATA.lbl = DATA.output;
    case 29,
        DATA.input = load('movingSquares.data')';
        DATA.output = load('movingSquares.labels')' + 1;
        DATA.lbl = DATA.output;
    case 30,
        DATA.input = load('transientChessboard.data')';
        DATA.output = load('transientChessboard.labels')' + 1;
        DATA.lbl = DATA.output;
    case 31,
        DATA.input = load('mixedDrift.data')';
        DATA.output = load('mixedDrift.labels')' + 1;
        DATA.lbl = DATA.output;
    case 32,
        loaded_data = load('ledDrift.txt')';
        DATA.input = loaded_data(1:end-1,:);
        DATA.output = loaded_data(end,:) + 1;
        DATA.lbl = DATA.output;
    case 33, % weather
        loaded_data = load('weather.mat');
        DATA.input = loaded_data.weather_data;
        DATA.output = loaded_data.weather_class;
        DATA.lbl = DATA.output;
    case 34, % electricity
        loaded_data = load('elec_market.mat');
        DATA.input = loaded_data.elec2data';
        DATA.output = loaded_data.elec2label';
        DATA.lbl = DATA.output;
    case 35, % Cover Type
        loaded_data = load('covtypeNorm.txt');
        DATA.input = loaded_data(:,1:54)';
        DATA.output = loaded_data(:,55)';
        DATA.lbl = DATA.output;
    case 36, % Poker
        DATA.input = load('poker.data')';
        DATA.output = load('poker.labels')';
        DATA.lbl = DATA.output;
    case 37, % Outdoor
        DATA.input = load('outdoorStream.data')';
        DATA.output = load('outdoorStream.labels')';
        DATA.lbl = DATA.output;
    case 38 % Rialto
        DATA.input = load('rialto.data')';
        DATA.output = load('rialto.labels')';
        DATA.lbl = DATA.output;
    case 39 % Spam        
        DATA.input = load('spam.data')';
        DATA.output = load('spam.labels')';
        DATA.lbl = DATA.output;
    otherwise
        % None of the sets
        disp('Unknown Data Base. Void Structure Created')
end

%% FILL OUTPUT STRUCTURE

DATAout = DATA;

%% END