function plot_data_pairplot(DATA,label,mode)

% --- Histograms and Pairwise Scatter from Input Variables ---
% 
%   plot_data_pairplot(DATA,label,mode)
%
%   Inputs:
%       DATA.
%           input = input matrix                        [p x N]
%           output = output matrix                      [1 x N]
%
%       label = cell with the names of the attributes   [1 x p]
%       mode = how the pairplot will be
%           'histogram'
%           'bar'
%           'cdf'
%           'both'
%   
%   P.S.: Adapted from: Ryosuke Takeuchi 2016/12/22 - 2017-01-24

%% INITIALIZATIONS

X = DATA.input';                    % Input matrix
[~,p] = size(X);                    % Number of attributes
if (p > 10),                        % Restrict number of attributes
    p = 10;
end

Y = DATA.output';                   % output matrix
[~,Nc] = size(Y);
if(Nc == 1),
    Nclasses = length(unique(Y));	% Number of classes
else
    Nclasses = Nc;
    [~,Y] = max(Y,[],2);
end

colors = lines(length(unique(Y)));  % colors for scatter plots

%% ALGORITHM

% Begin Figure

figure;
hold on

if nargin < 3
    mode = 'bar';
end

if nargin < 2
    label = cell(p,1);
    for i = 1:p,
        label{i} = int2str(i);
    end
end

for i = 1:p
    for j = 1:p
        
        subplot(p,p,sub2ind([p p], i, j));
        
        if i == 1
            ylabel(label{j},'fontweight','bold');
        end
        
        if j == p
            xlabel(label{i},'fontweight','bold');
        end
        
        hold on;
        
        if i == j
            bin = linspace(min(X(:,i)), max(X(:,i)), 20);
            for c = 1:Nclasses
                switch mode
                    case 'bar'
                        [counts,~] = histc(X(Y == c, i), bin);
                        bar(bin, counts, 'BarWidth', 1, 'FaceColor', colors(c,:))
                        xlim([bin(1) bin(end)]);
                    case 'histogram'
                        histogram(X(Y == c, i), bin, 'FaceColor', colors(c,:), ...
                                  'Normalization', 'probability');
                        xlim([bin(1) bin(end)])
                    case 'cdf'
                        [f, x] = ecdf(X(Y == c, i));
                        plot(x, f, 'Color', colors(c,:));
                end
            end
        else
            for c = 1:Nclasses
                plot(X(Y == c, i), X(Y == c, j), '.', 'Color', colors(c,:));
            end
            xlim([min(X(:, i)) max(X(:, i))])
        end
    end
end

% Finish Figure

hold off

%% END