%% UNIFORM DISTRIBUTED DATA

clear;
clc;

N = 1000;
dados1 = rand(2,N);

figure;
plot(dados1(1,:),dados1(2,:),'k.')

n = 0;

for i = 1:N,
    if ~((dados1(1,i) > 0.4 && dados1(1,i) < 0.6) || (dados1(2,i) > 0.4 && dados1(2,i) < 0.6))
        n = n+1;
        dados2(:,n) = dados1(:,i);
    end
end

figure;
plot(dados2(1,:),dados2(2,:),'k.')

dados = dados2;
alvos = zeros(1,length(dados2));

for i = 1:length(dados),
    if (dados(1,i) < 0.4 && dados(2,i) < 0.4)
        alvos(i) = 1;
    elseif (dados(1,i) < 0.4 && dados(2,i) > 0.6)
        alvos(i) = 2;
    elseif (dados(1,i) > 0.6 && dados(2,i) < 0.4)
        alvos(i) = 3;
    elseif (dados(1,i) > 0.6 && dados(2,i) > 0.6)
        alvos(i) = 4;
    end
end

figure;
hold on
samples = find(alvos == 1);
plot(dados2(1,samples),dados2(2,samples),'k.')
samples = find(alvos == 2);
plot(dados2(1,samples),dados2(2,samples),'y.')
samples = find(alvos == 3);
plot(dados2(1,samples),dados2(2,samples),'b.')
samples = find(alvos == 4);
plot(dados2(1,samples),dados2(2,samples),'r.')
hold off

savefile = 'data_four_groups.mat';
save(savefile, 'DATA');

%% END