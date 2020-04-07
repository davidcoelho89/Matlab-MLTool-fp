function [] = plot_stats_ssqe(ssqe_curve)

% --- Plot SSQE from clusters training ---
%
%   [] = plot_stats_SSE(sse_curve)
%
%   Input:
%       DATA.
%           input = input matrix            [p x N]
%           output = output matrix          [c x N]
%       PAR.
%           W = weight's matrix             [c x p+1]
%   Output:
%       frame = struct containing 'cdata' and 'colormap'

%% INITIALIZATION

% Get parameters

N = length(ssqe_curve);
min_ssqe = min(ssqe_curve);
max_ssqe = max(ssqe_curve);

% Init auxiliary variables

xaxis = 1:N;

%% ALGORITHM

figure;
hold on
title ('SSQE Curve');
xlabel('Epochs');
ylabel('SSQE');
axis ([0 N min_ssqe-0.1 max_ssqe+0.1]);
plot(xaxis,ssqe_curve);
hold off

%% END