% Apaga tudo que estava no workspace
clear; close; clc;

% Inicializa contagem de tempo
tic

% Qual Data Base; Tarefa; kernel
data = 7; task = 'ident'; kernel = 'GAUS';

% Carrega entradas e saidas
[u,y] = load_data(task,data); 

% Seleciona os hiperparâmetros ótimos para cada conjunto de dados
[ptr,C_aux,sig2_aux,n_u,n_y] = lssvr_parameters(task,data,kernel); 
[N_pad,C,sig2] = fs_lssvr_parameters(task,data);

% Se kernel Gaussiano, define hyperparametros
if strcmp(kernel,'GAUS'),
    par_1 = sig2;
    par_2 = 1; 
end

%% TREINO / TESTE

%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%

[linD,colD] = size(u);

J = round(ptr*linD);

if data == 75,
    % dados para treino da sequencia de entrada e saida
    treino_u_aux = u(J+1:end,:); 
    treino_y_aux = y(J+1:end,:); 
    % dados para teste da sequencia de entrada e saida
    teste_u_aux = u(1:J,:); 
    teste_y_aux = y(1:J,:); 
else
    % dados para treino da sequencia de entrada e saida
    treino_u_aux = u(1:J,:); 
    treino_y_aux = y(1:J,:); 
    % dados para teste da sequencia de entrada e saida
    teste_u_aux = u(J+1:end,:); 
    teste_y_aux = y(J+1:end,:); 
end

% Not normalized

treino_u = treino_u_aux;
treino_y = treino_y_aux;

teste_u = teste_u_aux;
teste_y = teste_y_aux;

[Ltr,Ctr] = size(treino_u);
[Lte,Cte] = size(teste_u);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Montar a matriz Reg_est_lssvr de regressores:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_ord = max(n_u,n_y);

vec_regres_tr = regres_mat(treino_u,treino_y,n_u,n_y);

treino_u1 = treino_u(n_ord+1:end,:);
treino_y1 = treino_y(n_ord+1:end,:);
treino_y1_plot = treino_y1;
treino_y1_aux = treino_y(n_ord+1:end,:);
teste_u1 = teste_u(n_ord+1:end,:);
teste_y1 = teste_y(n_ord+1:end,:);
teste_y1_aux = teste_y(n_ord+1:end,:);

[Ltr1,Ctr1] = size(vec_regres_tr);

Nr = 1; % número de rodadas independentes

for r = 1:Nr,

display(r);

S = randperm(Ltr1);
vec_regres_tr = vec_regres_tr(S,:);
treino_y1 = treino_y1(S,:);
treino_u1 = treino_u1(S,:);
treino_y1_aux = treino_y1_aux(S,:);

% Definition of working set:
M = round(N_pad*Ltr1);
vec_regres_tr_min = vec_regres_tr(1:M,:);
S_ind = S(1:M);
S_ind_ini = S_ind;
Omega_min = kernel_mat(kernel,vec_regres_tr_min,par_1,par_2);
[Lmin,Cmin] = size(vec_regres_tr_min);
entrop = -log((1/M^2)*sum(sum(Omega_min)));
entrop_ini = entrop;

% Definition of the complement of working set:
vec_regres_tr_comp = vec_regres_tr(M+1:end,:);
S_ind_comp = S(M+1:end);
[Lcomp,Ccomp] = size(vec_regres_tr_comp);

N_iter = 100;
i = 1;

% Active selection of prototype vectors using quadratic Renyi entropy: 
while i <= N_iter, 
      entrop_aux = entrop;
      en(i) = entrop;
      Omega_min_aux = Omega_min;
      vec_regres_tr_min_aux = vec_regres_tr_min;
      nn = randperm(Lmin,1);
      mm = randperm(Lcomp,1);
      XX = ismember(vec_regres_tr_min,vec_regres_tr_comp(mm,:),'rows');
      if max(XX) == 0,
         vec_regres_tr_min(nn,:) = vec_regres_tr_comp(mm,:);
         S_ind_aux(nn) = S_ind(nn);
         S_ind(nn) = S_ind_comp(mm);
         Omega_min = kernel_mat(kernel,vec_regres_tr_min,par_1,par_2);
         entrop = -log((1/M^2)*sum(sum(Omega_min)));
         if entrop <= entrop_aux,
            vec_regres_tr_min = vec_regres_tr_min_aux;
            entrop = entrop_aux;
            S_ind(nn) = S_ind_aux(nn);
         end
      end
      i = i+1;
end

% Auto decomposição da matriz Omega reduzida M x M: 

mat_reg_aux = eye(M) * 1e-6;    % evita problemas de condicionamento

Omega_min = kernel_mat(kernel,vec_regres_tr_min,par_1,par_2) + mat_reg_aux;

[U1,D1] = eig(Omega_min);
vec_D1 = diag(D1);
vec_D = -sort(-vec_D1);
D = diag(vec_D);

for j=1:M,
    U(:,j) = U1(:,M-j+1);
end

% Método de Nystrom para aproximar a matriz de Gram:

tic
mat_fi_aux = [];
for j = 1:M,
    vec_2 = vec_regres_tr_min(j,:);
    for i = 1:Ltr1,
        vec_1 = vec_regres_tr(i,:);
        K_aux(j,i) = exp(-norm(vec_1-vec_2).^2/2*par_1); 
    end
end
toc

tic
for j = 1:M,
     mat_fi_aux(:,j) = (1/sqrt(D(j,j)))*U(:,j)'*K_aux; 
end
toc

% tic
% for j=1:M,
%     for i=1:Ltr1,
%         vec_1=vec_regres_tr(i,:);
%         soma=0;
%         for z=1:M,
%             vec_2=vec_regres_tr_min(z,:);
%             soma = soma + U(z,j)*exp(-norm(vec_1-vec_2).^2/2*par_1);
%         end
%     mat_fi_aux1(i,j)=(1/sqrt(D(j,j)))*soma;
%     end
% end
% toc

% Solução do sistema linear com bias:

mat_fi = mat_fi_aux;
A = [mat_fi'*mat_fi+(1/C)*eye(M) mat_fi'*ones(Ltr1,1);...
ones(Ltr1,1)'*mat_fi ones(Ltr1,1)'*ones(Ltr1,1)];
vec_b = [mat_fi'*treino_y1_aux;ones(Ltr1,1)'*treino_y1_aux];
vec_w = pinv(A)*vec_b;
alpha_aux = vec_w(1:end-1,:);
b0 = vec_w(end,:);

alpha = [];
for j = 1:M,
    alpha_1 = 0;
    for i = 1:M,
        alpha_1 = alpha_1 + alpha_aux(i,1)*(1/sqrt(D(i,i)))*U(j,i);
    end
    alpha(j,1) = alpha_1;
end

%% CALCULANDO SAÍDAS E ERROS COM OS DADOS DE TREINAMENTO:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y_tr(:,r) = kernel_type(kernel,vec_regres_tr_min,vec_regres_tr,...
alpha,b0,par_1,par_2);

%% CALCULANDO SAÍDAS E ERROS COM OS DADOS DE TESTE:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(task,'regre'),
   vec_regres_te=regres_mat(teste_u,teste_y,n_u,n_y);
   Y_te(:,r)=kernel_type(kernel,vec_regres_tr_min,vec_regres_te,...
   alpha,b0,par_1,par_2); 
   
   teste_y1(:,r)=teste_y1_aux;
   treino_y1(:,r)=treino_y1_aux;
   [erro_tr,RMSE_tr(r)] = valida_modelos(treino_y1(:,r),Y_tr(:,r));
   [erro_te,RMSE_te(r)] = valida_modelos(teste_y1(:,r),Y_te(:,r));
else
    
option_teste = 2;

% (1) PREDIÇÃO 1 PASSO A FRENTE
% (2) SIMULAÇÃO LIVRE
% (3) PREDIÇÃO K PASSOS A FRENTE

k_ahead = 3;

% PREDIÇÃO DE 1 PASSO A FRENTE:
if option_teste == 1,
vec_regres_te = regres_mat(teste_u,teste_y,n_u,n_y);
Y_te1 = kernel_type(kernel,vec_regres_tr_min,vec_regres_te,...
alpha,b0,par_1,par_2);
%Y_te_lssvr = Y_te_lssvr1(n_ord+1:end)*di_y+mi_y;
Y_te(:,r) = Y_te1*di_y+mi_y;
%Y_te_lssvr = Y_te_lssvr1;

% SIMULAÇÃO LIVRE (PREDIÇÃO DE INFINITOS PASSOS A FRENTE):
elseif option_teste == 2,
[Y_te1] = simulacao_livre_regres(n_u,n_y,teste_u,teste_y,...
vec_regres_tr_min,kernel,par_1,par_2,alpha,b0);
Y_te(:,r) = Y_te1(n_ord+1:end)*di_y+mi_y;

% PREDIÇÃO DE k PASSOS A FRENTE):
elseif option_teste == 3,
[Y_te1] = predicao_kpassos(n_u,n_y,teste_u,teste_y,...
vec_regres_tr_min,kernel,par_1,par_2,alpha,b0,k_ahead);
Y_te(:,r) = Y_te1(n_ord+1:end)*di_y+mi_y;

end
teste_y1(:,r) = teste_y1_aux*di_y+mi_y;
treino_y1(:,r) = treino_y1_aux*di_y+mi_y;

[erro_tr,RMSE_tr(r)] = valida_modelos(treino_y1(:,r),Y_tr(:,r)*di_y+mi_y);
[erro_te,RMSE_te(r)] = valida_modelos(teste_y1(:,r),Y_te(:,r));
end

end % fim das rodadas independentes

[a,b] = max(RMSE_te);
Y_fs_lssvr = Y_te(:,b);

% PLOTANDO AS SAÍDAS:

if strcmp(task,'regre'),
   titulo_te = [sprintf('FS-LSSVR - Prediction (RMSE=%3.4f)',RMSE_te(b))]; 
else
    if option_teste == 1, 
    titulo_te = [sprintf('FS-LSSVR - Prediction (RMSE=%3.4f)',RMSE_te(b))];
    elseif option_teste == 2,
    titulo_te = [sprintf('FS-LSSVR - Simulation (RMSE=%3.4f)',RMSE_te(b))];
    elseif option_teste == 3,
    titulo_te = [sprintf('FS-LSSVR - Prediction%3.0f steps ahead(RMSE=%3.4f)',...
    k_ahead,RMSE_te(b))];
    end
end
[figure2] = gera_figuras(erro_tr,treino_u1,treino_y,...
Y_tr(:,r),teste_y1(:,b),Y_fs_lssvr,titulo_te);

% figure(2)
% boxplot(RMSE_te);
% hold on
% plot(mean(RMSE_te),'*');
% hold off
% ylabel('RMSE')

% ind_out_aux=[];
% for i=1:length(S_ind)
%     for j=1:length(ind_out)
%         if S_ind(i)==ind_out(j),
%            ind_out_aux=[ind_out_aux S_ind(i)];
%         end
%     end
% end
% figure(3)
% subplot(3,1,1);
% plot(treino_y1_plot,'b--');
% hold on
% plot(S_ind_ini,treino_y1_plot(S_ind_ini),'r*');
% title('INITIAL Prototype Vectors');
% hold off
% subplot(3,1,2);
% plot(treino_y1_plot,'b--');
% hold on
% plot(S_ind,treino_y1_plot(S_ind),'r*');
% title('FINAL Prototype Vectors');
% hold off
% subplot(3,1,3);
% plot(treino_y1_plot,'b--');
% hold on
% plot(ind_out_aux,treino_y1_plot(ind_out_aux),'g*');
% title('Outliers as Prototypes');
% %plot(8,treino_y1_plot(8,:),'g*');
% %plot(42,treino_y1_plot(42,:),'g*');
% hold off

figure(4)
plot(en,'b-');

toc
erro_medio = mean(RMSE_te)
erro_std = std(RMSE_te)

sort(S_ind)
