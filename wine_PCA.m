% 读入数据
data=xlsread("wine.csv");
class = data(:,1);
%% 对数据进行简单实验看看PCA 是否有效果
% 以index的前三个数据为实验
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
%% 不进行降维直接投影
figure 
scatter(data(:,2),data(:,3),20,C,'filled')
title('Direct projection without dimensionality reduction')
% 无数据标准化
[~,score_Test,~,~] = pca(data(:,2:4));
figure 
scatter(score_Test(:,1),score_Test(:,2),20,C,'filled')
title('Direct projection without data standard')
%% 对数据做PCA 变换
S_data = zscore(data(:,2:end));  % 数据标准化
r = corrcoef(S_data);            % 计算相关系数矩阵
[vec1,lamda,rate] = pcacov(r);   %相关系数矩阵进行主成分分析，vec1为r的特征值
contr = cumsum(rate); %计算积累贡献率
f = repmat(sign(sum(vec1)),size(vec1,1),1); %构造与vec1同维数的元素为±1的矩阵
vec2 = vec1.*f; %修改特征向量的正负号，使得每个特征向量的分量和为正
num = 8;  %选取的主成分的个数 （根据主成分的贡献，将数据降维成八维）
score = S_data*vec2(:,1:num);   %计算各个主成分的得分,score 即为降维之后的数据
C_score = score*rate(1:num)/100; %计算综合得分
[stf,ind] = sort(C_score,'descend');%把得分从高到低排列
% 以其中三个为例查看降维结果
figure 
scatter(stf,ind,20,C,'filled')
title('Dimensional reduction results')

%% 进行PCA降维处理
[pc,score_1,latent,tsquare] = pca(S_data);
cumsum(latent)./sum(latent)
% tran=pc(:,1:8);
tran=pc(:,1:2);
feature= bsxfun(@minus,S_data,mean(S_data,1));
feature_after_PCA= feature*tran;  %pc中对应每一维度对原始数据的精度:
figure 
scatter(feature_after_PCA(:,1),feature_after_PCA(:,2),20,C,'filled')
title('Dimensional reduction results in second method')
% 保存程序数据
data_PCA=[data(:,1) feature_after_PCA];
save ('data_PCA.mat','data_PCA');