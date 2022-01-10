% 读入数据
clear all
clc
data=xlsread("wine.csv");
% class = data(:,1);
class_mean=ones(3,14);
%% 分别求出期望值以及总的期望值
for i=1:3
    class_mean(i,:)=mean(data(data(:,1)==i,:));
end
% all_mean=mean(data_1);
%% 对每一类数据都有十三维，这里只取其中三维进行画图示意
% m=zeros(1,4);% 读入数据
% data=xlsread("wine.csv");
data_1=data(:,2:end);
class = data(:,1);
class_mean=ones(3,14);
%% 分别求出期望值以及总的期望值
for i=1:3
    class_mean(i,:)=mean(data(data(:,1)==i,:));
end
all_mean=mean(data_1);
%% 对每一类数据都有十三维，这里只取其中三维进行画图示意
m=zeros(1,4);
for i=2:4
   [m(i),~]=size( data(data(:,1)==i-1,:));
   m(i)=m(i-1)+m(i);
end
m(1)=1;
C=[ones(m(2),1);2*ones(m(3)-m(2),1);3*ones(m(4)-m(3),1)];
figure
scatter3(data(:,2),data(:,3),data(:,4),20,C*1.2,'filled');
title('Three-dimensional data without any process')
hold on
scatter3(class_mean(:,2),class_mean(:,3),class_mean(:,4),70,[1;2;3],'*')

%% 进行LDA 处理
% 计算类间离散度矩阵
x=ones(3,13);
sb=0;
newspace=zeros(13,2);
for i=1:3
    x(i,:)=all_mean-class_mean(i,2:end);
    sb=sb+m(i+1)*x(i,:)'*x(i,:)/178;
end
% 计算类内离散度矩阵
y=zeros(13,13,3);
sw=zeros(13,13);
for j = 1:3
    for i = m(j):m(j+1)
        y(:,:,j)=y(:,:,j)+(data_1(i,:)-class_mean(j))'*(data_1(i,:)-class_mean(j));
    end
    sw=sw+m(j+1)*y(:,:,j)/178;
end
% 求最大特征向量和特征向量;
[V,L]=eig(sw\sb);
[~,b]=max(max(L));
newspace(:,1)=V(:,b);            %最大特征值所对应的特征向量
L_I=max(L);
L_I(b)=[];
[~,b1]=max(L_I);
newspace(:,2)=V(:,b1);           %次大特征值所对应的特征向量
% new_class=ones(178,1);
%% 
new_class1=data_1(m(1):m(2),:)*newspace;               %第i类每个样本的降维结果放在对应的行中
new_class2=data_1(m(2)+1:m(3),:)*newspace;
new_class3=data_1(m(3)+1:m(4),:)*newspace;
%% 
figure
plot(new_class1(:,1),new_class1(:,2),'ro')
hold on 
plot(new_class2(:,1),new_class2(:,2),'b*')
hold on 
plot(new_class3(:,1),new_class3(:,2),'gs')
title('LDA-dimension')
data_LDA_whole=[new_class1;new_class2;new_class3];
data_LDA=[data(:,1) data_LDA_whole];
save ('data_LDA.mat','data_LDA');
%% KLDA 的算法实现 降为3维
Dim=3;          %dim为最后要降成的维数
redu_result=KLDA_proj_reduce(data(:,2:end), data(:,1), Dim);
%% 
[m,~]=size(data);
N1= data(data(:,1)==1,:);
N2= data(data(:,1)==2,:);
N3= data((data(:,1)==3),:);
[m1,~]=size(N1);
[m2,~]=size(N2);
[m3,~]=size(N3);
C=[ones(m1,1);2*ones(m2,1);3*ones(m3,1)];
figure 
scatter3(redu_result(:,1),redu_result(:,2),redu_result(:,3),20,C,'filled');
title('KLDA-3-dimension')
%% KLDA
Dim=2;          %dim为最后要降成的维数
redu_result_1=KLDA_proj_reduce(data(:,2:end), data(:,1), Dim);
%% 
figure
plot(redu_result_1(1:m1,1),redu_result_1(1:m1,2),'r*')
hold on 
plot(redu_result_1(m1+1:m2+m1,1),redu_result_1(m1+1:m2+m1,2),'go')
hold on 
plot(redu_result_1(m2+m1+1:m2+m1+m3,1),redu_result_1(m1+m2+1:m1+m2+m3,2),'bs')
title('KLDA-2-dimension,PolyPlus')
%% 保存数据
data_KLDA=[data(:,1) redu_result_1];
save ('data_KLDA.mat','data_KLDA');