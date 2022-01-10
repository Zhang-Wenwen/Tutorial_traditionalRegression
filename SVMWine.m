% %% I. 清空环境变量
clear all
clc
%% II. 导入数据,经KLDA,KPCA降维之后的数据
% load data_KPCA_new.mat 此数据已经是经过KPCA或者KLDA降维之后的数据
% load data_KPCA.mat %KPCA的SVM 数据
% load data_KLDA.mat   %KLDA的SVM数据
% data_KPCA=data_LDA;
% load data_PCA.mat
% data_KPCA=data_PCA;
load 'data_LDA.mat';
data_KPCA=data_LDA;
% 为了保证SVM能得到均匀分布的三类样本数据，将数组打乱,有助于提高分类的准确率
randIndex = randperm(size(data_KPCA,1));
data_KPCA_new=data_KPCA(randIndex,:);
%%
% % 2. 训练集――80%的样本

[P_m,P_n]=size(data_KPCA_new);
train_matrix = data_KPCA_new(1:floor(P_m*0.8),2:end);
train_label = data_KPCA_new(1:floor(P_m*0.8),1);
% % 3. 测试集――20%的样本
test_matrix = data_KPCA_new(floor(P_m*0.8):end,2:end);
test_label = data_KPCA_new(floor(P_m*0.8):end,1);
%  前80%做训练集，后20%做预测集

%% III. 数据归一化
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',test_matrix',PS);
Test_matrix = Test_matrix';
%% IV. SVM创建/训练(RBF核函数)
% 1. 寻找最佳c/g参数――交叉验证方法
[c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 0;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
        cg(i,j) = libsvmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
 
%%
% 2. 创建/训练SVM模型
model = libsvmtrain(train_label,Train_matrix,cmd);
% classes1=libsvmclassify(model,data_KPCA,'showplot',true);%data数据分类，并显示图形

%% 用于画图
N1= find(train_label(:,1)==1);
N2= find(train_label(:,1)==2);
N3= find(train_label(:,1)==3);
A=full(model.SVs);   %标出支持向量
if size(Train_matrix)==[142 2]
% Linea1=svmclassify(A);
figure
plot(Train_matrix(N1,1),Train_matrix(N1,2),'r*')
hold on 
plot(Train_matrix(N2,1),Train_matrix(N2,2),'go')
hold on 
plot(Train_matrix(N3,1),Train_matrix(N3,2),'yd')
hold on 
plot(A(:,1),A(:,2),'bs')
hold on
elseif size(Train_matrix)==[142 1]
figure
plot(Train_matrix(N1,1),'r*')
hold on 
plot(Train_matrix(N2,1),'go')
hold on 
plot(Train_matrix(N3,1),'yd')
hold on 
plot(A(:,1),'bs')
hold on
title('SVM')
end
%% V. SVM仿真测试
[predict_label_1,accuracy_1,decision_values1] = libsvmpredict(train_label,Train_matrix,model); 
[predict_label_2,accuracy_2,decision_values2] = libsvmpredict(test_label,Test_matrix,model); 
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];
%% VI. 绘图
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('真实类别','预测类别')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集SVM预测结果对比(RBF核函数)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)