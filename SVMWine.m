% %% I. ��ջ�������
clear all
clc
%% II. ��������,��KLDA,KPCA��ά֮�������
% load data_KPCA_new.mat �������Ѿ��Ǿ���KPCA����KLDA��ά֮�������
% load data_KPCA.mat %KPCA��SVM ����
% load data_KLDA.mat   %KLDA��SVM����
% data_KPCA=data_LDA;
% load data_PCA.mat
% data_KPCA=data_PCA;
load 'data_LDA.mat';
data_KPCA=data_LDA;
% Ϊ�˱�֤SVM�ܵõ����ȷֲ��������������ݣ����������,��������߷����׼ȷ��
randIndex = randperm(size(data_KPCA,1));
data_KPCA_new=data_KPCA(randIndex,:);
%%
% % 2. ѵ�����D�D80%������

[P_m,P_n]=size(data_KPCA_new);
train_matrix = data_KPCA_new(1:floor(P_m*0.8),2:end);
train_label = data_KPCA_new(1:floor(P_m*0.8),1);
% % 3. ���Լ��D�D20%������
test_matrix = data_KPCA_new(floor(P_m*0.8):end,2:end);
test_label = data_KPCA_new(floor(P_m*0.8):end,1);
%  ǰ80%��ѵ��������20%��Ԥ�⼯

%% III. ���ݹ�һ��
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',test_matrix',PS);
Test_matrix = Test_matrix';
%% IV. SVM����/ѵ��(RBF�˺���)
% 1. Ѱ�����c/g�����D�D������֤����
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
% 2. ����/ѵ��SVMģ��
model = libsvmtrain(train_label,Train_matrix,cmd);
% classes1=libsvmclassify(model,data_KPCA,'showplot',true);%data���ݷ��࣬����ʾͼ��

%% ���ڻ�ͼ
N1= find(train_label(:,1)==1);
N2= find(train_label(:,1)==2);
N3= find(train_label(:,1)==3);
A=full(model.SVs);   %���֧������
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
%% V. SVM�������
[predict_label_1,accuracy_1,decision_values1] = libsvmpredict(train_label,Train_matrix,model); 
[predict_label_2,accuracy_2,decision_values2] = libsvmpredict(test_label,Test_matrix,model); 
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];
%% VI. ��ͼ
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('��ʵ���','Ԥ�����')
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ�SVMԤ�����Ա�(RBF�˺���)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)