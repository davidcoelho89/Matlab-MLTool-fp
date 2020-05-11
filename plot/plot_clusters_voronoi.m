function [] = plot_clusters_voronoi(DATA,OUT_CL,OPTION)

% --- Plot cluster centroids, voronoi cells and Data ---
%
%   [] = plot_clusters_voronoi(DATA,OUT_CL,OPTION)
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

% Get chosen attributes
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

% Choose Grid length
grid_len = 100;

% Get max and min values of attributes
xaxis_min = min(X(Xaxis,:));
xaxis_max = max(X(Xaxis,:));
yaxis_min = min(X(Yaxis,:));
yaxis_max = max(X(Yaxis,:));

% Init auxiliary variables
xn = zeros(2,1);                        % 2d point sample
border_points = zeros(2,grid_len^2);	% 2d points of hyperplane
count = 0;                              % Count number of points from 
                                        %  voronoi cells borders

%% ALGORITHM

% Begin Figure

figure;
hold on

% Define figure properties

s1 = 'Attribute ';  s2 = int2str(Xaxis);    s3 = int2str(Yaxis);
xlabel(strcat(s1,s2));
ylabel(strcat(s1,s3));
title ('2D Voronoi Cells');

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

% ----- Plot prototypes

plot(Cx(Xaxis,:),Cx(Yaxis,:),'k*')

% ----- Plot Voronoi Borders

Cx = Cx([Xaxis,Yaxis],:);

% Define grid
xaxis_values = linspace(xaxis_min,xaxis_max,grid_len);
yaxis_values = linspace(yaxis_min,yaxis_max,grid_len);

% Get border points (horizontal)
for i = 1:grid_len,
    % Get first winner prototype of line
    xn(Xaxis) = xaxis_values(i);
    xn(Yaxis) = yaxis_values(1);
    win_previous = prototypes_win(Cx,xn,OUT_CL);
    for j = 2:grid_len,
        xn(Yaxis) = yaxis_values(j);
        win_current = prototypes_win(Cx,xn,OUT_CL);
        if (win_current ~= win_previous)
            count = count+1;
            border_points(:,count) = xn;
        end
        win_previous = win_current;
    end
end

% Get border points (vertical)
for i = 1:grid_len,
    % Get first winner prototype of column
    xn(Yaxis) = yaxis_values(i);
    xn(Xaxis) = xaxis_values(1);
    win_previous = prototypes_win(Cx,xn,OUT_CL);
    for j = 2:grid_len,
        xn(Xaxis) = xaxis_values(j);
        win_current = prototypes_win(Cx,xn,OUT_CL);
        if (win_current ~= win_previous)
            count = count+1;
            border_points(:,count) = xn;
        end
        win_previous = win_current;
    end
    
end

% Plot Border ( just get points that were saved)
border_points = border_points(:,1:count);
plot(border_points(1,:),border_points(2,:),'k.')
axis([xaxis_min-0.1,  xaxis_max+0.1,  yaxis_min-0.1,  yaxis_max+0.1])

% Finish Figure

hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END