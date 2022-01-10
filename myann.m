%读取训练数据
clear
clc
%------本代码采用ANN对酒的种类进行分类
%f1 f2 是2个特征值
% load data_KLDA.mat
% data=data_KLDA;
% load data_PCA.mat
% data=data_PCA;
load data_LDA.mat;
data=data_LDA;
% load data_KPCA.mat
% data=data_KPCA;
[P_m,~]=size(data);
randIndex = randperm(size(data,1));
data=data(randIndex,:); 
f2= data(1:floor(P_m*0.8),3);
f1=data(1:floor(P_m*0.8),2);
class=data(1:floor(P_m*0.8),1);
%特征值归一化
[input,minI,maxI] = premnmx( [f1 , f2]')  ;
%构造输出矩阵
s = length( class) ;
output = zeros( s , 3  ) ;
for i = 1 : s 
   output( i , class( i )  ) = 1 ;
end
%创建神经网络
net = newff( minmax(input) , [10 3] , { 'logsig' 'purelin' } , 'traingdx' ) ; 
%{
    minmax(input)：获取4个输入信号（存储在f1 f2中）的最大值和最小值；
    [10,3]：表示使用2层网络，第一层网络节点数为10，第二层网络节点数为3；
    { 'logsig' 'purelin' }：
        表示每一层相应神经元的激活函数；
        即：第一层神经元的激活函数为logsig（线性函数），第二层为purelin（对数S形转移函数）
    'traingdx'：表示学习规则采用的学习方法为traingdx（梯度下降自适应学习率训练函数）
%}
%设置训练參数
net.trainparam.show = 50 ;% 显示中间结果的周期
net.trainparam.epochs = 500 ;%	（学习次数）
net.trainparam.goal = 0.01 ;%神经网络训练的目标误差
net.trainParam.lr = 0.01 ;%学习速率（Learning rate）
%开始训练
%其中input为训练集的输入信号，对应output为训练集的输出结果
net = train( net, input , output' ) ;
%================================训练完成====================================%
%=============================接下来进行测试=================================%
 
%读取测试数据
% [t1 t2 c] = textread('testData.txt' , '%f%f%f%f%f',150);
 t1=data(floor(P_m*0.8)+1:end,2);
 t2=data(floor(P_m*0.8)+1:end,3);
 c=data(floor(P_m*0.8)+1:end,1);
%测试数据归一化
testInput = tramnmx ( [t1,t2]' , minI, maxI ) ;
%%
% testInput_1=[testInput;c'];
% p=net(testInput_1);
%[testInput,minI,maxI] = premnmx( [t1 , t2 ]')  ;
%仿真
%其中net为训练后得到的网络，返回的Y为
Y = sim( net , testInput )
% p=net(testInput);
%统计识别正确率
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
indexes=zeros(s2,1);  %用于记录最后预测后酒的种类的结果
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;
    indexes(i)=Index;
    if( Index  == c(i)   ) 
        hitNum = hitNum + 1 ; 
    end
end
sprintf('识别率是 %3.3f%%',100 * hitNum / s2 )

%% VI. 绘图
figure
plot(1:length(indexes),indexes,'r-*')
hold on
plot(1:length(indexes),c,'b:o')
grid on
legend('真实类别','预测类别')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集神经网络测试结果对比';
          ['accuracy = ' num2str(100 * hitNum / s2) '%']};
title(string)