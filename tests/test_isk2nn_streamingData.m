%% Machine Learning ToolBox

% isK2nn Model testing in various streaming datasets
% Author: David Nascimento Coelho
% Last Update: 2020/04/08

%% Datasets:

% # code: # samples / # attributes / # classes
% Brief Description

%% LED

% 32: 200k / 24 / 10
% Atributes = 0 or 1. Represents a 7 segments display.
% 17 Irrelevant Attributes. Which attribute is irrelevant: changes.
% Incremental Drift.

% test_led_isk2nn

%% Sea Concepts

% 25: 200k / 03 / 02
% label noise (10%)
% f1 + f2 = b; b is changing each 5000 samples.
% Abrupt drift

test_sea_isk2nn;

%% Rotating Hyperplane

% 200k x 10 x 02. 
% Moving Hyperplane. Gradual Drift.

test_hyper_isk2nn;

%% RBF Moving

% 200k x 10 x 05. 
% Moving RBFs. Different Mean. Gradual drift.

% test_rbfmov_isk2nn;

%% RBF Interchange

% 200k x 02 x 15. 
% Interchanging RBFs. Change Means. Abrupt drift.

test_rbfint_isk2nn;

%% Moving Squares

% 200k x 02 x 04. 
% Moving Squares. Gradual/Incremental drift.

% test_squmov_isk2nn;

%% Transient Chessboard

% 200k x 02 x 08. 
% Virtual Reocurring drifts.

% test_chess_isk2nn

%% Mixed Drift

% 600k x 02 x 15. 
% Various drifts.

% test_mixed_isk2nn;

%% Weather

% 18159 x 08 x 02
% Virtual Drift

test_weather_isk2nn;

%% Electricity

% 45312 x 08 x 02
% Real Drift

% test_electricity_isk2nn;

%% Cover Type

% 581012 x 54 x 07
% Real Drift

% test_covertype_isk2nn;

%% Poker Hand

% 829201 x 10 x 10
% Virtual Drift

% test_poker_isk2nn;

%% Outdoor

% 4000 x 21 x 40
% Virtual Drift

% test_outdoor_isk2nn;

%% Rialto

% 82250 x 27 x 10
% Virtual Drift

test_rialto_isk2nn;

%% Spam

% 
% 

%% END