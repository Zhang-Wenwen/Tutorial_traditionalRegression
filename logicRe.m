clear
clc
load data_KPCA.mat
% load data_KLDA.mat
% load data_PCA.mat
% load data_LDA.mat
%%
data=data_KPCA;
% data=data_KLDA;
% data=data_PCA;
% data=data_LDA;
%在data上加常数特征项；
data=[data,ones(size(data,1),1)];
[~,n]=size(data);
%% 数据归一化处理
data(:,2)=data(:,2)/max(data(:,2));
data(:,3)=data(:,3)/max(data(:,3));
%% 多分类问题改成三个两分类问题
N1= find(data(:,1)==1);
N2= find(data(:,1)==2);
N3= find(data(:,1)==3);
data_1=data(N1,:);
data_2=data(N2,:);
data_3=data(N3,:);

max_x=max(data(:,2));
max_y=max(data(:,3));
min_x=min(data(:,2));
min_y=min(data(:,3));

%% 两两分类 除了第一类都是负样本
data2_V1=[zeros(size(N2)) data_2(:,2:end)];   % 负样本
data3_V1=[zeros(size(N3)) data_3(:,2:end)];    % 负样本
data1_v1=[ones(size(N1)) data_1(:,2:end)];    % 正样本
data_12=[data1_v1;data2_V1;data3_V1];
[P_m,P_n]=size(data_12);
randIndex = randperm(size(data_12,1));
data_12=data_12(randIndex,:);
CLASS=data(randIndex,1);
CLASS_test=CLASS(floor(P_m*0.8)+1:end,1);
%区分训练集和预测集

% % 2. 训练集――80%的样本
Features=P_n-1;   %特征数量=维数-标签数量
train_matrix = data_12(1:floor(P_m*0.8),2:end);
train_label = data_12(1:floor(P_m*0.8),1);
% train_label(train_label(:,:)~=1)=0;  % 将所有标签不为1的分为0
% % 3. 测试集――20%的样本
test_matrix = data_12(floor(P_m*0.8)+1:end,2:end);
test_label = data_12(floor(P_m*0.8)+1:end,1);
% test_label(test_label(:,:)~=1)=0;
% 第一次分类的结果  第一次分类即分出第一类
output_1=zeros(floor(P_m*0.2)+1,1);
[acc,theta1,L,output_1]=logisticRegression(train_matrix,train_label,test_matrix,test_label,Features,output_1);
disp('预测第一类正确个数为为：')
disp(acc)
figure
subplot(1,2,1)
plot(L)
title('loss')

subplot(1,2,2)
x=min_x:0.05:max_x;
y=(-theta1(1)*x-theta1(3))/theta1(2);
plot(x,y,'linewidth',2)
hold on
plot(data(N1,2),data(N1,3),'r*')
hold on
plot(data(N2,2),data(N2,3),'go')
hold on
plot(data(N3,2),data(N3,3),'go')
hold on
title('第一次分类结果')
axis([min_x max_x min_y max_y])
%% 第二次分类的结果  将剩下两类（第二类，第三类）继续分类 除了第二类都是负样本
data2_V1=[ones(size(N2)) data_2(:,2:end)];   % 正样本
data3_V1=[zeros(size(N3)) data_3(:,2:end)];    % 负样本
data1_v1=[zeros(size(N1)) data_1(:,2:end)];    % 负样本
data_23=[data1_v1;data2_V1;data3_V1];
data_23=data_23(randIndex,:);
% 区分训练集和预测集

% % 2. 训练集――80%的样本
Features=P_n-1;   %特征数量=维数-标签数量
% train_matrix = data_23(1:floor(P_m*0.8),2:end);
train_label = data_23(1:floor(P_m*0.8),1);
% train_label(train_label(:,:)~=1)=0;  % 将所有标签不为1的分为-1
% % 3. 测试集――20%的样本
% test_matrix_1 = data_23(floor(P_m*0.8)+1:end,2:end);
test_label = data_23(floor(P_m*0.8)+1:end,1);
% 开始训练
output_2=zeros(floor(P_m*0.2)+1,1);
[acc_1,theta1_1,L_1,output_2]=logisticRegression(train_matrix,train_label,test_matrix,test_label,Features,output_2);
disp('预测第二类正确个数为为：')
disp(acc_1)
figure
subplot(1,2,1)
plot(L_1)
title('loss')

subplot(1,2,2)
x=min_x:0.05:max_x;
y_1=(-theta1_1(1)*x-theta1_1(3))/theta1_1(2);
plot(x,y_1,'linewidth',2)
hold on
plot(data(N2,2),data(N2,3),'yd')
hold on
plot(data(N3,2),data(N3,3),'go')
hold on

title('第二次分类结果')
axis([min_x max_x min_y max_y])
%% 第三次分类的结果  除了第三类都属负样本
data2_V1=[zeros(size(N2)) data_2(:,2:end)];   % 负样本
data3_V1=[ones(size(N3)) data_3(:,2:end)];    % 正样本
data1_V1=[zeros(size(N1)) data_1(:,2:end)];    % 负样本
data_13=[data1_V1;data2_V1;data3_V1];
data_13=data_13(randIndex,:);
% 区分训练集和预测集

% % 2. 训练集――80%的样本
[P_m,P_n]=size(data_13);
Features=P_n-1;   %特征数量=维数-标签数量
% train_matrix = data_13(1:floor(P_m*0.8),2:end);
train_label = data_13(1:floor(P_m*0.8),1);
% train_label(train_label(:,:)~=1)=0;  % 将所有标签不为1的分为-1
% % 3. 测试集――20%的样本
% test_matrix_1 = data_13(floor(P_m*0.8):end,2:end);
test_label = data_13(floor(P_m*0.8)+1:end,1);
% 开始训练
output_3=zeros(floor(P_m*0.2)+1,1);
[acc_3,theta1_3,L_3,output_3]=logisticRegression(train_matrix,train_label,test_matrix,test_label,Features,output_3);
disp('预测第三类正确个数为为：')
disp(acc_3)
figure
subplot(1,2,1)
plot(L_3)
title('loss')

subplot(1,2,2)
x=min_x:0.05:max_x;
y_3=(-theta1_3(1)*x-theta1_3(3))/theta1_3(2);
plot(x,y_3,'linewidth',2)
hold on
plot(data(N2,2),data(N2,3),'yo')
hold on
plot(data(N3,2),data(N3,3),'bd')
hold on

title('第三次分类结果')
axis([min_x max_x min_y max_y])
%% 总的预测正确率以及结果图
output=output_1;
output((output_2==1))=2+output((output_2==1));
output((output_3==1))=3+output((output_3==1));
err=sum(abs(output-CLASS_test))/(P_m*0.8);
sprintf('预测的总正确率：%f',1-err)
[output_1 output_2 output_3]
%%
figure
plot(data(N2,2),data(N2,3),'r*')
hold on
plot(data(N3,2),data(N3,3),'go')
hold on
plot(data(N1,2),data(N1,3),'bd')
hold on
plot(x,y_1,'linewidth',2)
hold on
% plot(x(I:end),y(I:end),'linewidth',2)
plot(x,y,'linewidth',2)
hold on
plot(x,y_3,'linewidth',2)
title('总的分类结果')
axis([min_x max_x min_y max_y])