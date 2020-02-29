%BAYES
for z = 1:Nr,
    
figure; plot(comp_bay{z}(:,1),comp_bay{z}(:,2),'k.')
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Rotulo')                % label eixo y
xlabel('Saida da Rede')         % label eixo x
title('Tendency Analysis')      % Titulo
axis ([-1.1 1.1 -1.1 1.1])      % Eixos

end

%OLS
for z = 1:Nr,
    
figure; plot(comp_ols{z}(:,1),comp_ols{z}(:,2),'k.')
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Rotulo')                % label eixo y
xlabel('Saida da Rede')         % label eixo x
title('Tendency Analysis')      % Titulo
axis ([-1.1 1.1 -1.1 1.1])      % Eixos

end

problem = prob;           %problema igual ao definido anteriormente
rejection = 0.4;          %qual será a faixa de rejeiçao
Mconf_r = zeros(7,7,Nr);  %matriz de confusao de cada repeticao

%MLP
for z = 1:Nr,
    
figure; plot(comp_mlp{z}(:,1),comp_mlp{z}(:,2),'k.')
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Rotulo')                % label eixo y
xlabel('Saida da Rede')         % label eixo x
title('Tendency Analysis')      % Titulo
axis ([-1.1 1.1 -1.1 1.1])      % Eixos

[Mconf,rejected] = reject_opt(comp_mlp{z},rejection,problem);

Mconf_r(:,:,z) = Mconf;
rejected_r{z} = rejected; 

end

Mconf_r(:,:,2),
rejected_r{2},

accuracy = (sum(sum(Mconf_r(:,:,2))) - sum(Mconf_r(:,1,2)) ...
 - sum(Mconf_r(1,:,2)) + 2*Mconf_r(1,1,2)) / sum(sum(Mconf_r(:,:,2)))

%ELM
for z = 1:Nr,
    
figure; plot(comp_elm{z}(:,1),comp_elm{z}(:,2),'k.')
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Rotulo')                % label eixo y
xlabel('Saida da Rede')         % label eixo x
title('Tendency Analysis')      % Titulo
axis ([-1.1 1.1 -1.1 1.1])      % Eixos

end

%MLM
for z = 1:Nr,
    
figure; plot(comp_mlm{z}(:,1),comp_mlm{z}(:,2),'k.')
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Rotulo')                % label eixo y
xlabel('Saida da Rede')         % label eixo x
title('Tendency Analysis')      % Titulo
axis ([-1.1 1.1 -1.1 1.1])      % Eixos

end

%SVM
for z = 1:Nr,
    
figure; plot(comp_svm{z}(:,1),comp_svm{z}(:,2),'k.')
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Rotulo')                % label eixo y
xlabel('Saida da Rede')         % label eixo x
title('Tendency Analysis')      % Titulo
axis ([-1.1 1.1 -1.1 1.1])      % Eixos

end

%LSSVM
for z = 1:Nr,
    
figure; plot(comp_lssvm{z}(:,1),comp_lssvm{z}(:,2),'k.')
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Rotulo')                % label eixo y
xlabel('Saida da Rede')         % label eixo x
title('Tendency Analysis')      % Titulo
axis ([-1.1 1.1 -1.1 1.1])      % Eixos

end
