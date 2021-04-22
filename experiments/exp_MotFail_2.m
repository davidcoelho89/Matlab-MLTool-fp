%% Machine Learning ToolBox

% Classification Iterations
% Author: David Nascimento Coelho
% Last Update: 2019/11/29

format long e;  % Output data style (float)

%% TESTE 1

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT =  struct('prob',7,'prob2',1,'norm',2,'lbl',0,'Nr',50, ...
              'hold',1,'ptrn',0.8,'file','teste01.mat'); 
CVp =  struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 2

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',1,'norm',3,'lbl',0,'Nr',50, ...
             'hold',1,'ptrn',0.8,'file','teste02.mat'); 
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 3

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',1,'norm',2,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste03.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 4

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',1,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste04.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 5

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',2,'norm',2,'lbl',0,'Nr',50, ...
             'hold',1,'ptrn',0.8,'file','teste05.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 6

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',2,'norm',3,'lbl',0,'Nr',50, ...
             'hold',1,'ptrn',0.8,'file','teste06.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 7

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',2,'norm',2,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste07.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 8

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',2,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste08.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 9

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',3,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste09.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 10

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',4,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste10.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 11

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',5,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste11.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 12

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',6,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste12.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 13

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',7,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste13.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 14

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',8,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste14.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 15

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',1,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste15.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% TESTE 16

close all;                      % Close all windows
clear;                          % Clear all variables
clc;                            % Clear command window

OPT = struct('prob',7,'prob2',2,'norm',3,'lbl',0,'Nr',50, ...
             'hold',2,'ptrn',0.8,'file','teste16.mat');
CVp = struct('fold',5);
REJp = struct('band',0.3,'w',0.25);

test_MotFail_3(OPT,CVp,REJp);    % calls class iterations

%% END