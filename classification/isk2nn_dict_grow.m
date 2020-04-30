function [PAR] = isk2nn_dict_grow(DATA,HP)

% --- Sparsification Procedure for Dictionary Grow ---
%
%   [PAR] = isk2nn_dict_grow(DATA,HP)
%
%   Input:
%       DATA.
%           xt = attributes of sample                           [p x 1]
%           yt = class of sample                                [Nc x 1]
%       HP.
%           Cx = Attributes of input dictionary                 [p x Nk]
%           Cy = Classes of input dictionary                    [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]
%           Dm = Design Method                                  [cte]
%               = 1 -> all data set
%               = 2 -> per class
%           Ss = Sparsification strategy                        [cte]
%               = 1 -> ALD
%               = 2 -> Coherence
%               = 3 -> Novelty
%               = 4 -> Surprise
%           v1 = Sparseness parameter 1                         [cte]
%           v2 = Sparseness parameter 2                         [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%       OUT.
%           y_h = classifier's output                           [Nc x 1]
%           win = closest prototype to sample                   [1 x 1]
%           dist = distance of sample from each prototype       [Nk x 1]
%   Output: 
%       PAR.
%           Cx = Attributes of output dictionary                [p x Nk]
%           Cy = Classes of  output dictionary                  [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]

%% INITIALIZATIONS

% Get Hyperparameters

Dm = HP.Dm;                         % Design method
Ss = HP.Ss;                         % Sparsification Strategy
v1 = HP.v1;                         % Sparsification parameter 1
% v2 = PAR.v2;                      % Sparsification parameter 2
sig2n = HP.sig2n;                   % Kernel regularization parameter

% Get Parameters

Dx = HP.Cx;                         % Attributes of dictionary
Dy = HP.Cy;                         % Classes of dictionary
Km = HP.Km;                         % Dictionary Kernel Matrix (total)
Kmc = HP.Kmc;                       % Dictionary Kernel Matrix (class)
Kinv = HP.Kinv;                     % Dictionary Inv Kernel Matrix (total)
Kinvc = HP.Kinvc;                   % Dictionary Inv Kernel Matrix (class)
score = HP.score;                   % Prototypes score for prunning
class_history = HP.class_history;	% Prototypes last classification
times_selected = HP.times_selected; % Prototypes # of selection

% Get Data

xt = DATA.input;
yt = DATA.output;

% Get problem parameters

[~,m] = size(Dx);   % hold dictionary size

%% 1 DICTIONARY FOR ALL DATA SET

if Dm == 1, 
    
    % First Element of dictionary
    if (m == 0),
        
        Dx_out = xt;
        Dy_out = yt;
        Km_out = kernel_func(xt,xt,HP) + sig2n;
        Kinv_out = 1/Km_out;
        score_out = 0;
        class_history_out = 0;
        times_selected_out = 0;
        
	% ALD Criterion
    elseif Ss == 1,
        
        % Calculate kt
        kt_c = zeros(m,1);
        for i = 1:m,
            kt_c(i) = kernel_func(Dx(:,i),xt,HP);
        end
        
        % Calculate ktt
        ktt = kernel_func(xt,xt,HP);
        
        % Calculate ald coefficients
        at = Kinv*kt_c;
        
        % Calculate delta
        delta = ktt - kt_c'*at;
        
        % "Normalized delta" => Avoid Conditioning problems
        delta = delta + sig2n;
        
        % Expand dictionary
        if (delta > v1),
            Dx_out = [Dx, xt];
            Dy_out = [Dy, yt];
            Km_out = [Km, kt_c; kt_c', ktt + sig2n];
            Kinv_out = (1/delta)*[delta*Kinv + at*at', -at; -at', 1];
            score_out = [score,0];
            class_history_out = [class_history,0];
            times_selected_out = [times_selected,0];
        % Do not expand dictionary
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_history_out = class_history;
            times_selected_out = times_selected;
        end
        
	% Coherence Criterion
    elseif Ss == 2,
        
        % Init coherence measure (first element of dictionary)
        u = kernel_func(Dx(:,1),xt,HP) / ...
            (sqrt(kernel_func(Dx(:,1),Dx(:,1),HP) * ...
            kernel_func(xt,xt,HP)));
        u_max = abs(u);
        
        % get coherence measure
        if (m >= 2),
            for i = 2:m,
                % Calculate kernel
                u = kernel_func(Dx(:,i),xt,HP) / ...
                    (sqrt(kernel_func(Dx(:,i),Dx(:,i),HP) * ...
                    kernel_func(xt,xt,HP)));
                % Calculate Coherence
                if (abs(u) > u_max),
                    u_max = abs(u);
                end
            end
        end
        
        % Expand dictionary
        if (u_max <= v1),
            Dx_out = [Dx, xt];
            Dy_out = [Dy, yt];
            Km_out = Km;            % ToDo - Update if used to
            Kinv_out = Kinv;        % build other models!
            score_out = [score,0];
            class_history_out = [class_history,0];
            times_selected_out = [times_selected,0];
        % Do not expand dictionary
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_history_out = class_history;
            times_selected_out = times_selected;
        end
        
	% Novelty Criterion
    elseif Ss == 3,
        
        % Find nearest prototype
        win = prototypes_win(Dx,xt,HP);
        
        % Calculate distance from nearest prototype
        dist1 = vectors_dist(Dx(:,win),xt,HP);
        
        % Novelty conditions
        if(dist1 > v1),
            HP.Cx = Dx; HP.Cy = Dy;               % get current dict
            DATA.input = xt;                        % get current input
            OUT = prototypes_class(DATA,HP);       % estimate output

%             % Expand dictionary if estimation and real output are very
%             % diferrent from each other
%             dist2 = vectors_dist(yt,OUT.y_h,PAR);
%             if (dist2 > v2),

            % Expand dictionary if the sample was missclassified 
            [~,yh_seq] = max(OUT.y_h); 
            [~,yt_seq] = max(yt);
            if (yt_seq ~= yh_seq),
                Dx_out = [Dx, xt];
                Dy_out = [Dy, yt];
                Km_out = Km;            % ToDo - Update if used to 
                Kinv_out = Kinv;        % build other models!
                score_out = [score,0];
                class_history_out = [class_history,0];
                times_selected_out = [times_selected,0];
            % Do not expand dictionary
            else
                Dx_out = Dx;
                Dy_out = Dy;
                Km_out = Km;
                Kinv_out = Kinv;
                score_out = score;
                class_history_out = class_history;
                times_selected_out = times_selected;
            end
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_history_out = class_history;
            times_selected_out = times_selected;
        end
        
	% Surprise Criterion
    elseif Ss == 4,
        
        % Calculate h(t) (same as k(t) from ALD)
        ht_c = zeros(m,1);
        for i = 1:m,
            ht_c(i) = kernel_func(Dx(:,i),xt,HP);
        end
        
        % Estimated output ( y_h = ( ht' / Gt ) * Dy' ) (from GP)
        y_h = (ht_c' * Kinv) * Dy';
        
        % Calculate ktt
        ktt = kernel_func(xt,xt,HP);
        
        % Estimated variance ( sig2 = sig2n + ktt - ( ht' / Gt ) * ht )
        at = Kinv * ht_c;
        sig2 = sig2n + ktt -  ht_c' * at;

        % Surprise measure
        Si = log(sqrt(sig2)) + (norm(y_h' - yt,2)^2) / (2 * sig2);

        % Expand dictionary
        if (Si >= v1),
            Dx_out = [Dx, xt];
            Dy_out = [Dy, yt];
            Km_out = [Km, ht_c; ht_c', ktt + sig2n];
            Kinv_out = (1/sig2)*[sig2*Kinv + at*at', -at; -at', 1];
            score_out = [score,0];
            class_history_out = [class_history,0];
            times_selected_out = [times_selected,0];
        % Do not expand dictionary
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_history_out = class_history;
            times_selected_out = times_selected;
        end
        
    end
    
end

%% 1 DICTIONARY FOR EACH CLASS

if Dm == 2,
    
    % First Element of all dictionaries
    if (m == 0),

        % Add samples to dictionary
        Dx_out = xt;
        Dy_out = yt;
        % Get number of classes and class of sample
        [Nc,~] = size(yt);
        [~,c] = max(yt);
        % Build Kernel matrix and its inverse of class
        Km_out = cell(Nc,1);
        Km_out{c} = kernel_func(xt,xt,HP) + sig2n;
        Kinv_out = cell(Nc,1);
        Kinv_out{c} = 1/Km_out{c};
        % Init Scores
        score_out = 0;
        class_history_out = 0;
        times_selected_out = 0;
    else
        
        % Get sample class and dictionary labels in sequential pattern
        [~,c] = max(yt);           	% get sequential class of sample
        [~,Dy_seq] = max(Dy);   	% get sequential classes of dictionary
        mc = sum(Dy_seq == c);      % number of prototypes from class c
        
        % First Element of a class dictionary
        if (mc == 0),

            % Add sample to dictionary
            Dx_out = [Dx, xt];
            Dy_out = [Dy, yt];
            % Build Kernel matrix and its inverse of class
            Km{c} = kernel_func(xt,xt,HP) + sig2n;
            Km_out = Km;
            Kinv{c} = 1/Km{c};
            Kinv_out = Kinv;
            % Add score
            score_out = [score,0];
            class_history_out = [class_history,0];
            times_selected_out = [times_selected,0];
        else
            
            % Get inputs and outputs from class c
            Dx_c = Dx(:,Dy_seq == c);
            Dy_c = Dy(:,Dy_seq == c);
            
            % Get Kernel and Inverse Kernel Matrix
            Km_c = Km{c};
            Kinv_c = Kinv{c};
            
            % ALD Method
            if Ss == 1,
                
                % Calculate kt
                kt_c = zeros(mc,1);
                for i = 1:mc,
                    kt_c(i) = kernel_func(Dx_c(:,i),xt,HP);
                end
                
                % Calculate ktt
                ktt = kernel_func(xt,xt,HP);
                
                % Calculate coefficients
                at_c = Kinv_c*kt_c;
                
                % Calculate delta
                delta = ktt - kt_c'*at_c;
                
                % "Normalized delta" => avoid conditioning problems
                delta = delta + sig2n;
%                 display(delta); % debug

                % Expand dictionary
                if (delta > v1),
                    % Add sample to dictionary
                    Dx_out = [Dx, xt];
                    Dy_out = [Dy, yt];
                    % Kernel Matrix of class
                    Km{c} = [Km_c, kt_c; kt_c', ktt + sig2n];
                    Km_out = Km;
                    % Inverse Kernel Matrix of class
                    Kinv{c} = (1/delta)* ...
                              [delta*Kinv_c + at_c*at_c',-at_c;-at_c',1];
                    Kinv_out = Kinv;
                    % Add score of new prototype
                    score_out = [score,0];
                    class_history_out = [class_history,0];
                    times_selected_out = [times_selected,0];
                % Do not expand dictionary
                else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_history_out = class_history;
                    times_selected_out = times_selected;
                end
                
            % Coherence Method
            elseif Ss == 2,
                
                % init coherence measure
                u = kernel_func(Dx_c(:,1),xt,HP) / ...
                    (sqrt(kernel_func(Dx_c(:,1),Dx_c(:,1),HP) * ...
                    kernel_func(xt,xt,HP)));
                u_max = abs(u);
                
                % get coherence measure
                if (mc >= 2),
                    for i = 2:mc,
                        % Calculate kernel
                        u = kernel_func(Dx_c(:,i),xt,HP) / ...
                            (sqrt(kernel_func(Dx_c(:,i),Dx_c(:,i),HP) * ...
                             kernel_func(xt,xt,HP)));
                        % Calculate Coherence
                        if (abs(u) > u_max),
                            u_max = abs(u);
                        end
                    end
                end
                
                % Expand dictionary
                if (u_max <= v1),
                    Dx_out = [Dx, xt];
                    Dy_out = [Dy, yt];
                    Km_out = Km;            % ToDo - Update if used to 
                    Kinv_out = Kinv;        % build other models!
                    score_out = [score,0];
                    class_history_out = [class_history,0];
                    times_selected_out = [times_selected,0];
               % Do not expand dictionary
               else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_history_out = class_history;
                    times_selected_out = times_selected;
                end
                
            % Novelty
            elseif Ss == 3,
                
                % Find nearest prototype of class
                win = prototypes_win(Dx_c,xt,HP);
                
                % Calculate distance from nearest prototype
                dist1 = vectors_dist(Dx_c(:,win),xt,HP);
                
                % Novelty conditions
                if(dist1 > v1),
                    HP.Cx = Dx; HP.Cy = Dy;         % get current dict
                    DATA.input = xt;                  % get current input
                    OUT = prototypes_class(DATA,HP); % get class output

%                     % Expand dictionary if estimation and real output
%                     % are very diferrent from each other
%                     dist2 = vectors_dist(yt,OUT.y_h,PAR);
%                     if (dist2 > v2),
                    
                    % Expand dictionary if the sample was missclassified
                    [~,yt_seq] = max(yt);
                    [~,yh_seq] = max(OUT.y_h);
                    if (yt_seq ~= yh_seq),
                        Dx_out = [Dx, xt];
                        Dy_out = [Dy, yt];
                        Km_out = Km;            % ToDo - Update if used to 
                        Kinv_out = Kinv;        % build other models!
                        score_out = [score,0];
                        class_history_out = [class_history,0];
                        times_selected_out = [times_selected,0];
                    % Do not expand dictionary
                    else
                        Dx_out = Dx;
                        Dy_out = Dy;
                        Km_out = Km;
                        Kinv_out = Kinv;
                        score_out = score;
                        class_history_out = class_history;
                        times_selected_out = times_selected;
                    end
                % Do not expand dictionary
                else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_history_out = class_history;
                    times_selected_out = times_selected;
                end
                
            % Surprise
            elseif Ss == 4,
                
                % Calculate h(t) (same as k(t) from ALD)
                ht_c = zeros(mc,1);
                for i = 1:mc,
                    ht_c(i) = kernel_func(Dx_c(:,i),xt,HP);
                end
                
                % Estimated output ( y_h = ( ht' / Gt ) * Dy' ) (from GP)
                y_h = ( ht_c' * Kinv_c ) * Dy_c';
                
                % Calculate ktt
                ktt = kernel_func(xt,xt,HP);
                
                % Estimated variance (sig2 = sig2n + ktt - (ht' / Gt )* ht)
                at_c = Kinv_c * ht_c;
                sig2 = sig2n + ktt - ht_c' * at_c;

                % Surprise measure
                Si = log(sqrt(sig2)) + (norm(y_h' - yt,2)^2) / (2 * sig2);
                
                % Expand dictionary
                if (Si >= v1),
                    % Add sample to dictionary
                    Dx_out = [Dx, xt];
                    Dy_out = [Dy, yt];
                    % Kernel Matrix of class
                    Km{c} = [Km_c, ht_c; ht_c', ktt + sig2n];
                    Km_out = Km;
                    % Inverse Kernel Matrix of class
                    Kinv{c} = (1/sig2)*[sig2*Kinv_c + at_c*at_c',-at_c;-at_c',1];
                    Kinv_out = Kinv;
                    % Add score of new prototype
                    score_out = [score,0];
                    class_history_out = [class_history,0];
                    times_selected_out = [times_selected,0];
                % Do not expand dictionary
                else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_history_out = class_history;
                    times_selected_out = [times_selected,0];
                end
                
            end % end of Ss
            
        end % end of mc == 0
        
    end % end of m == 0
    
end % end of Dm == 2
    
%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = Dx_out;
PAR.Cy = Dy_out;
PAR.Km = Km_out;
PAR.Kmc = Kmc;
PAR.Kinv = Kinv_out;
PAR.Kinvc = Kinvc;
PAR.score = score_out;
PAR.class_history = class_history_out;
PAR.times_selected = times_selected_out;

%% END