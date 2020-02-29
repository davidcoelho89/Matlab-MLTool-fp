function [] = plot_labeled_neurons(PAR)

% --- Plot Neurons Grid with labels ---
%
%   [] = plot_labeled_neurons(PAR)
%
%   Input:
%       PAR.
%           Cx = prototypes                                     [p x Nk]
%           Cy = class of each prototype                        [Nc x Nk]
%           ind = indexes indicating each sample's cluster    	[1 x N]
%           SSE = squared error of each turn of training        [1 x Nep]
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get Clusters labels
Cy = PAR.Cy;

% Get grid dimensions and number of classes
[Nc,Nk] = size(Cy);
[~,labels] = max(Cy);

% Main types of markers, line style and colors

if Nc > 7,
    color_array = cell(1,Nc+1);
    color_array(1:7) = {'y','m','c','r','g','b','k'};
    color_array(Nc+1) = {'w'}; %last one is white
    for i = 8:Nc,
        color_array(i) = {rand(1,3)};
    end
else
    color_array = {'r','m','c','y','g','b','k','w'};
end

marker_array = {'.','*','o','x','+','s','d','v','^','<','>','p','h'};

%% ALGORITHM

% Begin Figure

figure;
hold on

if (isfield(PAR,'R')),
    R = PAR.R;
    [Ndim,~] = size(R);
    if (Ndim == 1),
        axis ([0 Nk+1 -1 +1]);
        for i = 1:Nk,
            if labels(i) ~= 0,
                plot_color = color_array{labels(i)};
                line_style =  marker_array{labels(i)};
                marker = strcat(plot_color,line_style);
                plot (i,0,marker);
            end
        end
    elseif (Ndim == 2),
        dim_len1 = max(R(1,:));
        dim_len2 = max(R(2,:));
        axis([0 dim_len1+1 0 dim_len2+1])
        for i = 1:Nk,
            if labels(i) ~= 0,
                plot_color = color_array{labels(i)};
                line_style =  marker_array{labels(i)};
                marker = strcat(plot_color,line_style);
                plot (R(1,i),R(2,i),marker);
            end            
        end
    elseif (Ndim == 3),
        % ToDo - Tridimensional Plot
    end
else
    axis ([0 Nk+1 -1 +1]);
    for i = 1:Nk,
        if labels(i) ~= 0,
            plot_color = color_array{labels(i)};
            line_style =  marker_array{labels(i)};
            marker = strcat(plot_color,line_style);
            plot (i,0,marker);
        end
    end
end

% Finish Figure

hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END