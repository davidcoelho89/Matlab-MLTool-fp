%% OOP Matlab

clear;
close;
clc;

%% INTRO - Classes, Properties and Methods

% See Tutorial: https://www.youtube.com/watch?v=n9Q7AQOhttw

% Value        -> variable -> container (struct) -> properties
% command line -> script   -> function           -> methods

% Formal relationship between data and functions
% Details not exposed to users
% Hard to break

% Class: blueprint for creating objects. Properties and Methods.
% Object: specific instance of a class.
%   - it is case sensitive
% Attributes: data inside object
%   - encapsulation: control access, restrict modification. 
% Inheritance: Subclasses and Superclasses
%   - get characteristics of other classes.
%   - builds on proven code
%   Ex:
%   - classdef Engineer < Employee
%   - classdef TestEngineer < Engineer
% Matlab's class diagram viewer
%   - can see class hierarchies

%% Instatiate Object

blip1 = blip;
blip1.AoA = 4;
blip1.signal = 5;

blip1.identify();


%% END




















