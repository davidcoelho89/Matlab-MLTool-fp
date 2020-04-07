function [] = plot_clusters_neigborhood(DATA,OUT_CL,OPTION)

% --- Plot Clusters Grid and Data ---
%
%   [] = plot_clusters_neigborhood(DATA,OUT_CL,OPTION)
%
%   Input:
%       DATA.
%           input = input matrix                             	[p x N]
%       OUT_CL.
%           Cx = prototypes                                     [p x Nk]
%           R = prototypes' grid positions                      [Nd x Nk]
%           ind = indexes indicating each sample's cluster    	[Nd x N]
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
    Xaxis = OPTION.Xaxis;
    Yaxis = OPTION.Yaxis;
end

% Get Data
X = DATA.input;

% Get index and Clusters prototypes
index = OUT_CL.ind;
Cx = OUT_CL.Cx;

% Get number of prototypes
[~,Nk] = size(Cx);

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
title ('2D of Clusters Grid');

% Plot Data

for i = 1:Nk,

    % Define Color
    if i <= 6,
        plot_color = color_array{i};
    else
        plot_color = rand(1,3);
    end
    
    % Define Marker as the LineStyle
    marker = marker_array{1};

    % Get samples from especific cluster
    samples = find(index == i);
    
    % Plot samples
    plot(X(Xaxis,samples),...
         X(Yaxis,samples),...
         marker,'MarkerFaceColor',...
         plot_color)
end    

% Plot prototypes and grid

if (isfield(OUT_CL,'R')),
    R = OUT_CL.R;
    [Ndim,~] = size(R);
    if (Ndim == 1),
        plot(Cx(Xaxis,:),Cx(Yaxis,:),'-ms',...
        'LineWidth',1,...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','g',...
        'MarkerSize',5)
    elseif (Ndim == 2),
        for dim = 1:Ndim,
            dim_len = max(R(dim,:));
            for i = 1:dim_len,
                samples = find(R(dim,:) == i);
              	plot(Cx(Xaxis,samples),Cx(Yaxis,samples),'-ms',...
               	'LineWidth',1,...
               	'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5) 
            end
        end
    elseif (Ndim == 3),
        % ToDo - Tridimensional Plot
    end
    
else
	plot(Cx(Xaxis,:),Cx(Yaxis,:),'-ms',...
	'LineWidth',1,...
	'MarkerEdgeColor','k',...
	'MarkerFaceColor','g',...
	'MarkerSize',5)   
end

% Finish Figure

hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END