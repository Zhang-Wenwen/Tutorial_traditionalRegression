clear all
clc
% load data_KPCA.mat
load data_KLDA.mat
% load data_PCA.mat
% load data_LDA.mat
%% 
% data=data_KPCA;
data=data_KLDA;
% data=data_PCA;
% data=data_LDA;
[m,n]=size(data);
N1= find(data(:,1)==1);
N2= find(data(:,1)==2);
N3= find(data(:,1)==3);
if n==3
figure
plot(data(N1,2),data(N1,3),'r*')
hold on 
plot(data(N2,2),data(N2,3),'go')
hold on 
plot(data(N3,2),data(N3,3),'yd')
hold on 
title('reduction data')
[center,U,objFcn]=FCMClust(data,3);
figure
plot(objFcn)
title('Objective Function Values')
xlabel('Iteration Count')
ylabel('Objective Function Value')
end 
if n==2
figure
plot(data(N1,2),'r*')
hold on 
plot(data(N2,2),'go')
hold on 
plot(data(N3,2),'yd')
hold on 
[center,U,objFcn]=FCMClust(data,3);
figure
plot(objFcn)
title('Objective Function Values')
xlabel('Iteration Count')
ylabel('Objective Function Value')
end 

%% 
maxU=max(U);
index1=find(U(1,:)==maxU);
index2=find(U(2,:)==maxU);
index3=find(U(3,:)==maxU);
%% 
if n==3
figure
% line(data(index1,2),data(index1,3),'linestyle','none','marker','o','color','g');
% line(data(index2,2),data(index2,3),'linestyle','none','marker','*','color','b');
% line(data(index3,2),data(index3,3),'linestyle','none','marker','d','color','r');
plot(data(index1,2),data(index1,3),'go');
hold on
plot(data(index2,2),data(index2,3),'b*');
hold on
plot(data(index3,2),data(index3,3),'dr');
hold on 
hold on 
plot(center(1,2),center(1,3),'ko','markersize',15,'LineWidth',2)
plot(center(2,2),center(2,3),'kx','markersize',15,'LineWidth',2)
plot(center(3,2),center(3,3),'ks','markersize',15,'LineWidth',2)
end
if n==2
    figure
% line(data(index1,2),data(index1,3),'linestyle','none','marker','o','color','g');
% line(data(index2,2),data(index2,3),'linestyle','none','marker','*','color','b');
% line(data(index3,2),data(index3,3),'linestyle','none','marker','d','color','r');
plot(data(index1,2),'go');
hold on
plot(data(index2,2),'b*');
hold on
plot(data(index3,2),'dr');
hold on 
hold on 
plot(center(1,2),'ko','markersize',15,'LineWidth',2)
plot(center(2,2),'kx','markersize',15,'LineWidth',2)
plot(center(3,2),'ks','markersize',15,'LineWidth',2)
end