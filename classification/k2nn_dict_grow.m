function [Dout] = k2nn_dict_grow(xt,yt,Din,PAR)

% --- Sparsification Procedure for Dictionary Grow ---
%
%   Dout = k2nn_dict_grow(xt,yt,Din,PAR)
%
%   Input:
%       xt = attributes of sample                               [p x 1]
%       yt = class of sample                                    [Nc x 1]
%       Din.
%           x = Attributes of input dictionary                  [p x Nk]
%           y = Classes of input dictionary                     [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           score = used for prunning method                    [1 x Nk]
%       PAR.
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
%   Output: 
%       Dout.
%           x = Attributes of output dictionary                 [p x Nk]
%           y = Classes of  output dictionary                   [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           score = used for prunning method                    [1 x Nk]
%           class_hist = used for prunning method               [1 x Nk]

%% INITIALIZATIONS

% Get dictionary
Dx = Din.x;                     % Attributes of dictionary
Dy = Din.y;                     % Classes of dictionary
Km = Din.Km;                    % Dictionary Kernel Matrix
Kinv = Din.Kinv;                % Dictionary Inverse Kernel Matrix
score = Din.score;              % Prototypes score for prunning
class_hist = Din.class_hist;    % Prototypes last classification

% Get Hyperparameters
Dm = PAR.Dm;        % Design method
Ss = PAR.Ss;        % Sparsification Strategy
v1 = PAR.v1;     	% Sparsification parameter 1
% v2 = PAR.v2;    	% Sparsification parameter 2
sig2n = PAR.sig2n;  % Kernel regularization parameter

% Get problem parameters
[~,m] = size(Dx);   % hold dictionary size

%% 1 DICTIONARY FOR ALL DATA SET

if Dm == 1, 
    
    % First Element of dictionary
    if (m == 0),
        
        Dx_out = xt;
        Dy_out = yt;
        Km_out = kernel_func(xt,xt,PAR) + sig2n;
        Kinv_out = 1/Km_out;
        score_out = 0;
        class_hist_out = 0;
        
	% ALD Criterion
    elseif Ss == 1,
        
        % Calculate kt
        kt_c = zeros(m,1);
        for i = 1:m,
            kt_c(i) = kernel_func(Dx(:,i),xt,PAR);
        end
        
        % Calculate ktt
        ktt = kernel_func(xt,xt,PAR);
        
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
            class_hist_out = [class_hist,0];
        % Do not expand dictionary
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_hist_out = class_hist;
        end
        
	% Coherence Criterion
    elseif Ss == 2,
        
        % Init coherence measure (first element of dictionary)
        u = kernel_func(Dx(:,1),xt,PAR) / ...
            (sqrt(kernel_func(Dx(:,1),Dx(:,1),PAR) * ...
            kernel_func(xt,xt,PAR)));
        u_max = abs(u);
        
        % get coherence measure
        if (m >= 2),
            for i = 2:m,
                % Calculate kernel
                u = kernel_func(Dx(:,i),xt,PAR) / ...
                    (sqrt(kernel_func(Dx(:,i),Dx(:,i),PAR) * ...
                    kernel_func(xt,xt,PAR)));
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
            class_hist_out = [class_hist,0];
        % Do not expand dictionary
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_hist_out = class_hist;
        end
        
	% Novelty Criterion
    elseif Ss == 3,
        
        % Find nearest prototype
        win = prototypes_win(Dx,xt,PAR);
        
        % Calculate distance from nearest prototype
        dist1 = vectors_dist(Dx(:,win),xt,PAR);
        
        % Novelty conditions
        if(dist1 > v1),
            PAR.Cx = Dx; PAR.Cy = Dy;               % get current dict
            DATA.input = xt;                        % get current input
            OUT = prototypes_class(DATA,PAR);       % estimate output

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
                class_hist_out = [class_hist,0];
            % Do not expand dictionary
            else
                Dx_out = Dx;
                Dy_out = Dy;
                Km_out = Km;
                Kinv_out = Kinv;
                score_out = score;
                class_hist_out = class_hist;
            end
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_hist_out = class_hist;
        end
        
	% Surprise Criterion
    elseif Ss == 4,
        
        % Calculate h(t) (same as k(t) from ALD)
        ht_c = zeros(m,1);
        for i = 1:m,
            ht_c(i) = kernel_func(Dx(:,i),xt,PAR);
        end
        
        % Estimated output ( y_h = ( ht' / Gt ) * Dy' ) (from GP)
        y_h = (ht_c' * Kinv) * Dy';
        
        % Calculate ktt
        ktt = kernel_func(xt,xt,PAR);
        
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
            class_hist_out = [class_hist,0];
        % Do not expand dictionary
        else
            Dx_out = Dx;
            Dy_out = Dy;
            Km_out = Km;
            Kinv_out = Kinv;
            score_out = score;
            class_hist_out = class_hist;
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
        Km_out{c} = kernel_func(xt,xt,PAR) + sig2n;
        Kinv_out = cell(Nc,1);
        Kinv_out{c} = 1/Km_out{c};
        % Init Scores
        score_out = 0;
        class_hist_out = 0;
        
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
            Km{c} = kernel_func(xt,xt,PAR) + sig2n;
            Km_out = Km;
            Kinv{c} = 1/Km{c};
            Kinv_out = Kinv;
            % Add score
            score_out = [score,0];
            class_hist_out = [class_hist,0];
            
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
                    kt_c(i) = kernel_func(Dx_c(:,i),xt,PAR);
                end
                
                % Calculate ktt
                ktt = kernel_func(xt,xt,PAR);
                
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
                    Kinv{c} = (1/delta)*[delta*Kinv_c + at_c*at_c',-at_c;-at_c',1];
                    Kinv_out = Kinv;
                    % Add score of new prototype
                    score_out = [score,0];
                    class_hist_out = [class_hist,0];
                % Do not expand dictionary
                else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_hist_out = class_hist;
                end
                
            % Coherence Method
            elseif Ss == 2,
                
                % init coherence measure
                u = kernel_func(Dx_c(:,1),xt,PAR) / ...
                    (sqrt(kernel_func(Dx_c(:,1),Dx_c(:,1),PAR) * ...
                    kernel_func(xt,xt,PAR)));
                u_max = abs(u);
                
                % get coherence measure
                if (mc >= 2),
                    for i = 2:mc,
                        % Calculate kernel
                        u = kernel_func(Dx_c(:,i),xt,PAR) / ...
                            (sqrt(kernel_func(Dx_c(:,i),Dx_c(:,i),PAR) * ...
                            kernel_func(xt,xt,PAR)));
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
                    class_hist_out = [class_hist,0];
               % Do not expand dictionary
               else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_hist_out = class_hist;
                end
                
            % Novelty
            elseif Ss == 3,
                
                % Find nearest prototype of class
                win = prototypes_win(Dx_c,xt,PAR);
                
                % Calculate distance from nearest prototype
                dist1 = vectors_dist(Dx_c(:,win),xt,PAR);
                
                % Novelty conditions
                if(dist1 > v1),
                    PAR.Cx = Dx; PAR.Cy = Dy;         % get current dict
                    DATA.input = xt;                  % get current input
                    OUT = prototypes_class(DATA,PAR); % get class output

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
                        class_hist_out = [class_hist,0];
                    % Do not expand dictionary
                    else
                        Dx_out = Dx;
                        Dy_out = Dy;
                        Km_out = Km;
                        Kinv_out = Kinv;
                        score_out = score;
                        class_hist_out = class_hist;
                    end
                % Do not expand dictionary
                else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_hist_out = class_hist;
                end
                
            % Surprise
            elseif Ss == 4,
                
                % Calculate h(t) (same as k(t) from ALD)
                ht_c = zeros(mc,1);
                for i = 1:mc,
                    ht_c(i) = kernel_func(Dx_c(:,i),xt,PAR);
                end
                
                % Estimated output ( y_h = ( ht' / Gt ) * Dy' ) (from GP)
                y_h = ( ht_c' * Kinv_c ) * Dy_c';
                
                % Calculate ktt
                ktt = kernel_func(xt,xt,PAR);
                
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
                    class_hist_out = [class_hist,0];
                % Do not expand dictionary
                else
                    Dx_out = Dx;
                    Dy_out = Dy;
                    Km_out = Km;
                    Kinv_out = Kinv;
                    score_out = score;
                    class_hist_out = class_hist;
                end
                
            end % end of Ss
            
        end % end of mc == 0
        
    end % end of m == 0
    
end % end of Dm == 2
    
%% FILL OUTPUT STRUCTURE

Dout = Din;
Dout.x = Dx_out;
Dout.y = Dy_out;
Dout.Km = Km_out;
Dout.Kinv = Kinv_out;
Dout.score = score_out;
Dout.class_hist = class_hist_out;

%% END