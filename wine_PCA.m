% ��������
data=xlsread("wine.csv");
class = data(:,1);
%% �����ݽ��м�ʵ�鿴��PCA �Ƿ���Ч��
% ��index��ǰ��������Ϊʵ��
[m,~]=size(data);
N1= data(data(:,1)==1,:);
N2= data(data(:,1)==2,:);
N3= data((data(:,1)==3),:);
[m1,~]=size(N1);
[m2,~]=size(N2);
[m3,~]=size(N3);
C=[ones(m1,1);2*ones(m2,1);3*ones(m3,1)];
figure
scatter3(data(:,2),data(:,3),data(:,4),20,C,'filled');
title('Three-dimensional validation data')
%% �����н�άֱ��ͶӰ
figure 
scatter(data(:,2),data(:,3),20,C,'filled')
title('Direct projection without dimensionality reduction')
% �����ݱ�׼��
[~,score_Test,~,~] = pca(data(:,2:4));
figure 
scatter(score_Test(:,1),score_Test(:,2),20,C,'filled')
title('Direct projection without data standard')
%% ��������PCA �任
S_data = zscore(data(:,2:end));  % ���ݱ�׼��
r = corrcoef(S_data);            % �������ϵ������
[vec1,lamda,rate] = pcacov(r);   %���ϵ������������ɷַ�����vec1Ϊr������ֵ
contr = cumsum(rate); %������۹�����
f = repmat(sign(sum(vec1)),size(vec1,1),1); %������vec1ͬά����Ԫ��Ϊ��1�ľ���
vec2 = vec1.*f; %�޸����������������ţ�ʹ��ÿ�����������ķ�����Ϊ��
num = 8;  %ѡȡ�����ɷֵĸ��� ���������ɷֵĹ��ף������ݽ�ά�ɰ�ά��
score = S_data*vec2(:,1:num);   %����������ɷֵĵ÷�,score ��Ϊ��ά֮�������
C_score = score*rate(1:num)/100; %�����ۺϵ÷�
[stf,ind] = sort(C_score,'descend');%�ѵ÷ִӸߵ�������
% ����������Ϊ���鿴��ά���
figure 
scatter(stf,ind,20,C,'filled')
title('Dimensional reduction results')

%% ����PCA��ά����
[pc,score_1,latent,tsquare] = pca(S_data);
cumsum(latent)./sum(latent)
% tran=pc(:,1:8);
tran=pc(:,1:2);
feature= bsxfun(@minus,S_data,mean(S_data,1));
feature_after_PCA= feature*tran;  %pc�ж�Ӧÿһά�ȶ�ԭʼ���ݵľ���:
figure 
scatter(feature_after_PCA(:,1),feature_after_PCA(:,2),20,C,'filled')
title('Dimensional reduction results in second method')
% �����������
data_PCA=[data(:,1) feature_after_PCA];
save ('data_PCA.mat','data_PCA');