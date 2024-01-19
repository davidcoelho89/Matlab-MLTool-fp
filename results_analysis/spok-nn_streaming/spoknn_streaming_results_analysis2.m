
% Prototypes
figure;
hold on 
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
title('Prototypes Location')
xlabel('Attribute 1')
ylabel('Attribute 2')
hold off

% Percentage of Misclassified
figure;
hold on
plot(x,1-accuracy_vector,'r-');
title('Percentage of samples misclassified')
xlabel('Time Step')
ylabel('Error Rate')
axis([-1 length(x) -0.02 0.1])
hold off

% Data and Prototypes
figure;
hold on 
plot(DATAttt.input(1,:),DATAttt.input(2,:),'r.');
plot(PAR.Cx(1,:),PAR.Cx(2,:),'k*');
title('Data and Prototypes')
xlabel('Attribute 1')
ylabel('Attribute 2')
hold off

% Number of Prototypes (Total)
figure;
colors = lines(Nc+1);
hold on
plot(x,prot_per_class(Nc+1,:),'Color',colors(Nc+1,:));
% for c = 1:Nc+1
%     plot(x,prot_per_class(c,:),'Color',colors(c,:));
% end
title('Number of prototypes per step')
xlabel('Steps')
ylabel('Number of prototypes')
hold off

prot_per_class1 = prot_per_class';
save('electricity.txt', 'prot_per_class1', '-ascii');