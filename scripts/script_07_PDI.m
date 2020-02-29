%% TRATAMENTO DE IMAGENS - MATLAB

% David Nascimento Coelho
% Última Revisão: 23/05/2014

clear all; close all; clc;

nfig = 0;       % contagem do numero de figuras

%% CARREGAR E PLOTAR IMAGENS

a = imread('data_Foto1.jpg');

r = a(:,:,1);
g = a(:,:,2);
b = a(:,:,3);

imshow(a);

%% OUTRAS

% Im2col();
% col2im();
% fliplr();
% flipud();

%% END