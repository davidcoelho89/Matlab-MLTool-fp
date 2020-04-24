function [PAR] = isk2nn_dict_prun(HP)

% --- Sparsification Procedure for Dictionary Pruning ---
%
%   [PAR] = isk2nn_dict_prun(HP)
%
%   Input:
%       HP.
%           Cx = Attributes of input dictionary                 [p x Nk]
%           Cy = Classes of input dictionary                    [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           score = used for prunning method                    [1 x Nk]
%           class_hist = used for prunning method               [1 x Nk]
%           Dm = Design Method                                  [cte]
%               = 1 -> all data set
%               = 2 -> per class
%           Ss = Sparsification strategy                        [cte]
%               = 1 -> ALD
%               = 2 -> Coherence
%               = 3 -> Novelty
%               = 4 -> Surprise
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method
%           min_score = score that leads to prune prototype     [cte]
%   Output: 
%       PAR.
%           Cx = Attributes of output dictionary                [p x Nk]
%           Cy = Classes of  output dictionary                  [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           score = score of each prototype from dictionary     [1 x Nk]

%% INITIALIZATIONS

% Get Hyperparameters
Dx = HP.Cx;               	% Attributes of dictionary
Dy = HP.Cy;              	% Classes of dictionary
Km = HP.Km;                	% Dictionary Kernel Matrix
Kinv = HP.Kinv;           	% Dictionary Inverse Kernel Matrix
score = HP.score;         	% Score of each prototype
class_hist = HP.class_hist;	% Classification history of each prototype
Dm = HP.Dm;               	% Design Method
Ss = HP.Ss;               	% Sparsification strategy
Ps = HP.Ps;                	% Pruning Strategy
min_score = HP.min_score; 	% Score that leads the prototype to be pruned

% Get problem parameters
[~,m] = size(Dx);           % hold dictionary size

%% 1 DICTIONARY FOR ALL DATA SET

if(Dm == 1),
    
    if (Ps == 0),
        
        % Does nothing
        
    elseif (Ps == 1),
        
        [~,Dy_seq] = max(Dy);	% get sequential label of dictionary
        
        for k = 1:m,
            
            % class of prototype
            c = Dy_seq(k);

            % number of elements from the same class as the prototypes'
            mc = sum(Dy_seq == c);
            
            % dont rem element if it is the only element of its class
            if (mc == 1),
                continue;
            end
            
            % Remove element
            if (score(k) < min_score),
                
                % Remove Prototype and its score
                Dx(:,k) = [];
                Dy(:,k) = [];
                score(k) = [];
                class_hist(k) = [];
                
                % If ALD or Surprise method, update kernel matrix
                if (Ss == 1 || Ss == 4),
                    % Remove line and column from inverse kernel matrix
                    ep = zeros(m,1);
                    ep(k) = 1;
                    u = Km(:,k) - ep;
                    eq = zeros(m,1);
                    eq(k) = 1;
                    v = eq;
                    Kinv = Kinv + (Kinv * u)*(v' * Kinv) / ...
                               (1 - v' * Kinv * u);
                    Kinv(k,:) = [];
                    Kinv(:,k) = [];
                    % Remove line and column from kernel matrix
                    Km(k,:) = [];
                    Km(:,k) = [];
                end
                
                % Just remove one prototype per loop
                break;
                
            end
        end
        
    elseif( Ps == 2),
        
        for k = 1:m,
          	
            % Remove element
            if (score(k) < min_score),
                
                % Remove Prototype and its score
                Dx(:,k) = [];
                Dy(:,k) = [];
                score(k) = [];
                class_hist(k) = [];
                
                % If ALD or Surprise method, update kernel matrix
                if (Ss == 1 || Ss == 4),
                    % Remove line and column from inverse kernel matrix
                    ep = zeros(m,1);
                    ep(k) = 1;
                    u = Km(:,k) - ep;
                    eq = zeros(m,1);
                    eq(k) = 1;
                    v = eq;
                    Kinv = Kinv + (Kinv * u)*(v' * Kinv) / ...
                               (1 - v' * Kinv * u);
                    Kinv(k,:) = [];
                    Kinv(:,k) = [];
                    % Remove line and column from kernel matrix
                    Km(k,:) = [];
                    Km(:,k) = [];
                end
                
                % Just remove one prototype per loop
                break;
                
            end
        end
    end
    
end

%% 1 DICTIONARY FOR EACH CLASS

if(Dm == 2),
    
    if (Ps == 0),

        % Does nothing
    
    elseif (Ps == 1),
        
        [~,Dy_seq] = max(Dy);	% get sequential label of dictionary
        
        for k = 1:m,
            
            % class of prototype
            c = Dy_seq(k);
            
            % number of elements from the same class as of prototype
            mc = sum(Dy_seq == c);
            
            % dont rem element if it is the only element of its class
            if (mc == 1),
                continue;
            end
            
            % Remove element
            if (score(k) < min_score),
                
                % Get prototypes from the same class
                Dx_c = Dx(:,Dy_seq == c);
                
                % Find position of prototype between elements of same class
                win_c = prototypes_win(Dx_c,Dx(:,k),HP);
                
                % Remove Prototype and its score
                Dx(:,k) = [];
                Dy(:,k) = [];
                score(k) = [];
                class_hist(k) = [];
                
             	% If ALD or Surprise method, update kernel matrix
                if (Ss == 1 || Ss == 4),
                    % Remove line and column from inverse kernel matrix
                    ep = zeros(mc,1);
                    ep(win_c) = 1;
                    u = Km{c}(:,win_c) - ep;
                    eq = zeros(mc,1);
                    eq(win_c) = 1;
                    v = eq;
                    Kinv{c} = Kinv{c} + (Kinv{c}*u)*(v'*Kinv{c}) / ...
                                  (1 - v'*Kinv{c}*u);
                    Kinv{c}(win_c,:) = [];
                    Kinv{c}(:,win_c) = [];
                    % Remove line and column from kernel matrix
                    Km{c}(win_c,:) = [];
                    Km{c}(:,win_c) = [];
                end
                
                % Just remove one prototype per loop
                break;
                
            end
        end
        
    elseif(Ps == 2),
        
        [~,Dy_seq] = max(Dy);	% get sequential label of dictionary
        
        for k = 1:m,
            
            % class of prototype
            c = Dy_seq(k);
            
            % number of elements from the same class as of prototype
            mc = sum(Dy_seq == c);
            
            % dont rem element if it is the only element of its class
%             if (mc == 1),
%                 continue;
%             end
            
            % Remove element
            if (score(k) < min_score),
                
                % Get prototypes from the same class
                Dx_c = Dx(:,Dy_seq == c);
                
                % Find position of prototype between elements of same class
                win_c = prototypes_win(Dx_c,Dx(:,k),HP);
                
                % Remove Prototype and its score
                Dx(:,k) = [];
                Dy(:,k) = [];
                score(k) = [];
                class_hist(k) = [];
                
             	% If ALD or Surprise method, update kernel matrix
                if (Ss == 1 || Ss == 4),
                    % Remove line and column from inverse kernel matrix
                    ep = zeros(mc,1);
                    ep(win_c) = 1;
                    u = Km{c}(:,win_c) - ep;
                    eq = zeros(mc,1);
                    eq(win_c) = 1;
                    v = eq;
                    Kinv{c} = Kinv{c} + (Kinv{c}*u)*(v'*Kinv{c}) / ...
                                  (1 - v'*Kinv{c}*u);
                    Kinv{c}(win_c,:) = [];
                    Kinv{c}(:,win_c) = [];
                    % Remove line and column from kernel matrix
                    Km{c}(win_c,:) = [];
                    Km{c}(:,win_c) = [];
                end
                
                % Just remove one prototype per loop
                break;
                
            end
        end        

    end
    
end

%% FILL OUTPUT STRUCTURE

PAR = HP;
PAR.Cx = Dx;
PAR.Cy = Dy;
PAR.Km = Km;
PAR.Kinv = Kinv;
PAR.score = score;
PAR.class_hist = class_hist;

%% END