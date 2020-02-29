%%
% ===================================================================
%          DOUTORADO EM ENGENHARIA DE TELEINFORMÁTICA
%   KERNEL RECURSIVE LEAST-SQUARES ALGORITHM EM PROBLEMAS DE
% IDENTIFICAÇÃO DE SISTEMAS, PREDIÇÃO DE SÉRIES TEMPORAIS E REGRESSÃO
%             Aluno: José Daniel de Alencar Santos
%         
%                            09/06/2015
%                            KERNEL-RLS
%
% ===================================================================

clear all; close all; clc;

path(path,'../../../Funções_Dados') % caminho relativo para valida_modelos
path(path,'../../../Funções_Dados/identificação') % caminho relativo para valida_modelos
path(path,'../../../Funções_Dados/regressão') % caminho relativo para valida_modelos
path(path,'../../LSSVR') % caminho relativo para valida_modelos

%% INICIALIZACOES

tic
task = 'tempo';
data = 5;

% carrega na memória as sequências de entrada e saída

[u,y] = load_data(task,data); 

% seleciona os hiperparâmetros ótimos para cada conjunto de dados

kernel = 'GAUS';
[ptr,C,sig2_aux,n_u,n_y,gamma]=lssvr_parameters(task,data,kernel);
[ni_v,ni_v1,lambda_f,sig2] = os_lssvr_parameters(task,data);
lambda_f = 1;

const = 0.01;

if (strcmp(kernel,'GAUS')),
    par_1 = sig2;
    par_2 = 1;
elseif (strcmp(kernel,'KMOD')),
    par_1 = sig2;
    par_2 = gamma;
end

%% INICIO DA RODADA DE TREINAMENTO/TESTE

% DEFINIÇÃO DOS DADOS DE TREINAMENTO E TESTE:

[linD,colD] = size(u);
J = round(ptr*linD);

if data == 75,
   treino_u_aux = u(J+1:end,:); % 
   treino_y_aux = y(J+1:end,:); % 
   teste_u_aux = u(1:J,:);      % 
   teste_y_aux = y(1:J,:);      % 
else
    treino_u_aux = u(1:J,:);    % dados para treino da sequencia de entrada
    treino_y_aux = y(1:J,:);    % dados para treino da sequencia de saida
    teste_u_aux = u(J+1:end,:); % dados para teste da sequencia de entrada
    teste_y_aux = y(J+1:end,:); % dados para teste da sequencia de saida
end

%teste_u_aux = treino_u_aux; % dados para teste da sequencia de entrada
%teste_y_aux = treino_y_aux; % dados para teste da sequencia de saida

% COM NORMALIZAÇÃO DOS DADOS:

% [treino_u,mi_u,di_u] = normaliza_dados(treino_u_aux,0,0,1);
% [treino_y,mi_y,di_y] = normaliza_dados(treino_y_aux,0,0,1);

% teste_u = normaliza_dados(teste_u_aux,mi_u,di_u,1);
% teste_y = normaliza_dados(teste_y_aux,mi_y,di_y,1);

% SEM NORMALIZAÇÃO DOS DADOS:

treino_u = treino_u_aux;
treino_y = treino_y_aux;
mi_u = 0;
di_u = 1;

teste_u = teste_u_aux;
teste_y = teste_y_aux;
mi_y = 0;
di_y = 1;

[Ltr,Ctr] = size(treino_u);
[Lte,Cte] = size(teste_u);

% Montar a matriz Reg_est_lssvr de regressores:

n_ord = max(n_u,n_y);

vec_regres_tr = regres_mat(treino_u,treino_y,n_u,n_y);

treino_u1 = treino_u(n_ord+1:end,:);
treino_y1_aux = treino_y(n_ord+1:end,:);
teste_u1 = teste_u(n_ord+1:end,:);
teste_y1_aux = teste_y(n_ord+1:end,:);

[Ltr1,Ctr1]=size(vec_regres_tr);

% número de rodadas independentes
Nr = 20;
MSE_train_total = [];

%% KERNEL RLS ALGORITHM

for r = 1:Nr,

display(r);

S = randperm(Ltr1);
vec_regres_tr = vec_regres_tr(S,:);
treino_y1_aux = treino_y1_aux(S,:);

% Inicialização:
vec_k_til = [];
vec_ini = vec_regres_tr(1,:);
dict = vec_ini;
K_til = kernel_out(kernel,vec_ini,vec_ini,par_1,par_2) + const^2;
K_til_menos = 1 / K_til;
alpha = treino_y1_aux(1,:) / K_til;
alpha = 0;
mat_P = 1;
m = 1;
MSE_train(1) = treino_y1_aux(1,:).^2;
m_aux(r,1) = 1;
t_aux(r,1) = 1;
Y_tr_aux(1,1) = 0;

for t = 2:Ltr1,

    vec = vec_regres_tr(t,:);
    y = treino_y1_aux(t,:);

    for i = 1:m,
        vec_aux = dict(i,:);
        vec_k_til(i,:) = kernel_out(kernel,vec,vec_aux,par_1,par_2) + const^2;
    end
    
    % ALD test:
    vec_a = K_til_menos * vec_k_til;
    k_tt = kernel_out(kernel,vec,vec,par_1,par_2) + const^2;
    delta = k_tt-vec_k_til' * vec_a;

    if delta > ni_v, % add vec to dictionary 
       dict = [dict; vec];
       K_til_menos = (1/delta) * ...
       [delta*K_til_menos+vec_a*vec_a' -vec_a; -vec_a' 1]; 	% EQ. IV.14
       [L,C] = size(vec_a);  
       mat_P = [mat_P zeros(L,1); zeros(L,1)' 1]; 			% EQ. IV.15
       %Y_tr_aux(t,1) = vec_k_til' * alpha;
       alpha = [alpha - (vec_a/delta)*(y - vec_k_til' * alpha); ...
       1 / delta*(y - vec_k_til'*alpha)]; 					% EQ. IV.16
       m=m+1;
       %
    else % dictionary unchanged
       vec_q = (mat_P*vec_a)/(lambda_f + vec_a'*mat_P*vec_a);
       mat_P = (1/lambda_f)*(mat_P - (mat_P*vec_a*vec_a'*mat_P)/...
       (lambda_f+vec_a'*mat_P*vec_a)); 								% EQ. IV.12
       alpha = alpha + K_til_menos*vec_q*(y - vec_k_til'*alpha); 	% EQ. IV.13
       %Y_tr_aux(t,1)=vec_k_til'*alpha;
    end

% Calcular o MSE durante o treinamento:    
Y_tr_aux = kernel_type_nobias(kernel,dict,vec_regres_tr(1:t,:),alpha,0,par_1,par_2,const);
MSE_aux = (treino_y1_aux(1:t,:) - Y_tr_aux(1:t,1)).^2;
MSE_train(t) = sum(MSE_aux);

end

[LL,CC] = size(alpha);
n_sv(r) = LL;
MSE_train_total = [MSE_train_total; MSE_train];

%% CALCULANDO SAÍDAS E ERROS COM OS DADOS DE TREINAMENTO:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y_tr(:,r)=kernel_type_nobias(kernel,dict,vec_regres_tr,alpha,0,...
par_1,par_2,const);

%% CALCULANDO SAÍDAS E ERROS COM OS DADOS DE TESTE:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if task=='regre',
   vec_regres_te = regres_mat(teste_u,teste_y,n_u,n_y);
   Y_te(:,r) = kernel_type_nobias(kernel,dict,vec_regres_te,...
   alpha,0,par_1,par_2,const); 
   %
   teste_y1(:,r) = teste_y1_aux;
   treino_y1(:,r) = treino_y1_aux;
   [erro_tr,RMSE_tr(r)] = valida_modelos(treino_y1(:,r),Y_tr(:,r));
   [erro_te,RMSE_te(r)] = valida_modelos(teste_y1(:,r),Y_te(:,r));
else

option_teste=3;

% (1) PREDIÇÃO 1 PASSO A FRENTE
% (2) SIMULAÇÃO LIVRE
% (3) PREDIÇÃO K PASSOS A FRENTE

k_ahead = 18;

% PREDIÇÃO DE 1 PASSO A FRENTE:
if option_teste == 1,

vec_regres_te = regres_mat(teste_u,teste_y,n_u,n_y);
Y_te1 = kernel_type_nobias(kernel,dict,vec_regres_te,alpha,0,par_1,par_2,const);
Y_te(:,r) = Y_te1*di_y + mi_y;

% SIMULAÇÃO LIVRE (PREDIÇÃO DE INFINITOS PASSOS A FRENTE):
elseif option_teste == 2,
[Y_te1] = simulacao_livre_regres_nobias(n_u,n_y,teste_u,teste_y,dict,kernel,par_1,par_2,alpha,0,const);
Y_te(:,r) = Y_te1(n_ord+1:end)*di_y + mi_y;

% PREDIÇÃO DE K PASSOS A FRENTE:
elseif option_teste == 3,
[Y_te1] = predicao_kpassos(n_u,n_y,teste_u,teste_y,dict,kernel,par_1,par_2,alpha,0,k_ahead);
Y_te(:,r) = Y_te1(n_ord+1:end)*di_y+mi_y;

end

teste_y1(:,r) = teste_y1_aux*di_y + mi_y;
treino_y1(:,r) = treino_y1_aux*di_y + mi_y;

[erro_tr,RMSE_tr(r)] = valida_modelos(treino_y1(:,r),Y_tr(:,r)*di_y+mi_y);
[erro_te,RMSE_te(r)] = valida_modelos(teste_y1(:,r),Y_te(:,r));

end

end

[a,b] = max(RMSE_te);
Y_krls = Y_te(:,b);

% PLOTANDO AS SAÍDAS:

if task == 'regre',
   titulo_te = [sprintf('KRLS - Prediction (RMSE=%3.4f)',RMSE_te(b))]; 
else
    if option_teste == 1, 
    titulo_te = [sprintf('KRLS - Prediction (RMSE=%3.4f)',RMSE_te(b))];
    elseif option_teste == 2,
    titulo_te = [sprintf('KRLS - Simulation (RMSE=%3.4f)',RMSE_te(b))];
    elseif option_teste == 3,
    titulo_te =[ sprintf('KRLS - Prediction%3.0f steps ahead(RMSE=%3.4f)',k_ahead,RMSE_te(b))];
    end
end

[figure2] = gera_figuras(erro_tr,treino_u1,treino_y,...
Y_tr(:,r),teste_y1(:,b),Y_krls,titulo_te);

% figure(2)
% boxplot(RMSE_te);
% hold on
% plot(mean(RMSE_te),'*');
% hold off

toc
erro_medio=mean(RMSE_te)
erro_std=std(RMSE_te)
mean_SV=mean(n_sv)

% figure()
% plot(mean(MSE_train_total))
% figure()
% plot(10*log(mean(MSE_train_total)))
% MG=10*log(mean(MSE_train_total));

MSE_final=mean(mean(MSE_train_total));
MSE_final_aux=log(MSE_final);
[aic,bic]=aicbic(MSE_final_aux,round(mean(n_sv)),Ltr1)
%AIC=Ltr1*log(MSE_final/Ltr1)+2*mean(n_sv);
%BIC=Ltr1*log(MSE_final/Ltr1)+mean(n_sv)*log(Ltr1);
