function [] = plot_time_series(DATAts)

% --- Plot Time Series from AR and ARX problems ---
%
%   [] = plot_time_series(DATAts)
%
%   Input:
%       DATAts.
%           input = input time series   [Nu x N]
%           output = output time series [Ny x N]
%       PAR.
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get Output Signals
y_ts = DATAts.output;
[Ny,N] = size(y_ts);

% Get Input Signals
if (isfield(DATAts,'input'))
    u_ts = DATAts.input;
    [Nu,~] = size(u_ts);
else
    Nu = 0;
end

% Initialize Samples Vector
n = 1:N;

% Define number of columns and lines of subplots
if (Nu == 0)
    Ncolumns = 1;
    Nlines = Ny;
else
    Ncolumns = 2;
    Nlines = max([Ny,Nu]);
end

%% ALGORITHM

figure;

if (Ncolumns == 1)
    for i = 1:Nlines
        y_ts_i = y_ts(i,:);
        subplot(Nlines,Ncolumns,i);
        plot(n,y_ts_i,'r-')
        title(int2str(i))
        axis([min(n)-1,max(n)+1,min(y_ts_i)-0.1,max(y_ts_i)+0.1])
    end
else
    for i = 1:Nlines
        if(i <= Nu)
            u_ts_i = u_ts(i,:);
            subplot(Nlines,Ncolumns,2*(i-1)+1);
            plot(n,u_ts_i,'r-')
            title(strcat('input ',int2str(i)))
            axis([min(n)-1,max(n)+1,min(u_ts_i)-0.1,max(u_ts_i)+0.1])
        end
        if(i <= Ny)
            y_ts_i = y_ts(i,:);
            subplot(Nlines,Ncolumns,2*(i-1)+2);
            plot(n,y_ts_i,'r-')
            title(strcat('output ',int2str(i)))
            axis([min(n)-1,max(n)+1,min(y_ts_i)-0.1,max(y_ts_i)+0.1])
        end
            
    end
end

%% END




























