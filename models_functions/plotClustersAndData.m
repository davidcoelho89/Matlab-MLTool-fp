function [] = plotClustersAndData(dataset,model,OPTIONS)

% --- Plot clusters Organization and Data ---
%
%   [] = plotClustersAndData(DATA,OUT_CL,OPTION)
%
%   Input:
%       DATA.
%           input = input matrix                             	[p x N]
%       OUT_CL.
%           Cx = prototypes                                     [p x Nk]
%           ind = indexes indicating each sample's cluster    	[1 x N]
%       OPTION.
%           Xaxis = Attribute to be plotted at x axis           [cte]
%           Yaxis = Attribute to be plotted at y axis           [cte]
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get figure options
if (nargin == 2)
    Xaxis = 1;
    Yaxis = 2;
else
    Xaxis = OPTIONS.Xaxis;
    Yaxis = OPTIONS.Yaxis;
end

% Main types of colors and markers
color_array =  {'r','y','m','c','g','b','k','w'};
marker_array = {'.','*','o','x','+','s','d','v','^','<','>','p','h'};

%% ALGORITHM

% Begin Figure

figure;
hold on

% Define figure properties

s1 = 'Attribute ';  s2 = int2str(Xaxis);    s3 = int2str(Yaxis);
xlabel(strcat(s1,s2));
ylabel(strcat(s1,s3));
title ('2D of Clusters Distribution');

% Plot Data

[~,Nk] = size(model.Cx);
for i = 1:Nk

    % Define Color
    if i <= 6
        plot_color = color_array{i};
    else
        plot_color = rand(1,3);
    end
    
    % Define Marker as the LineStyle
    marker = marker_array{1};

    % Get samples from especific cluster
    samples = find(model.Yh == i);
    
    % Plot samples
    plot(dataset.input(Xaxis,samples),...
         dataset.input(Yaxis,samples),...
         marker,'MarkerFaceColor',...
         plot_color)
end    

% Plot prototypes

plot(model.Cx(Xaxis,:),model.Cx(Yaxis,:),'k*')

% Finish Figure

hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END