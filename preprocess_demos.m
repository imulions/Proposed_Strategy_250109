function [x0 , xT, Data, index] = preprocess_demos(demos,time,tol_cutting)
%
% This function preprocess raw data and put them in a format suitable for
% SEDS. The function computes the first time derivative of demonstrations,
% shift the final point of all demonstrations to the origin (this only
% simplify computation load in SEDS), and trim datas. The function can be
% called using: 
% 该函数对原始数据进行预处理，并将其转换为适合SEDS的格式。
% 该函数计算演示的第一次导数，将所有演示的最后一点移到原点(这只会简化SEDS中的计算负载)，并修剪数据。
% 可以使用以下方式调用该函数:
%
%          [x0 , xT, Data, index] = preprocess_demos(demos,time,tol_cutting)
%
% Inputs -----------------------------------------------------------------
%
%
%   o demos:   A variable containing all demonstrations (only
%              trajectories). The variable 'demos' should follow the
%              following format:
%              - demos{n}: d x T^n matrix representing the d dimensional
%                          trajectories. T^n is the number of datapoint in
%                          this demonstration (1 < n < N)
%    o demos: 包含所有演示(仅轨迹)的变量。
%             变量'demos'应该遵循以下格式:
%             - demos{n}: d x T^n矩阵，表示d维轨迹。
%                         T^n是本演示中数据点的个数(1 < n < N)
%
%   o time:    This variable can be provided in two ways. If time steps
%              between all demonstrations are the same, 'time' could be
%              given as a positive scalar (i.e. time = dt). If not, 'time'
%              should follow the following format:
%              - time{n}: 1 x T^n vector representing the time array of length
%                         T^n corresponding to the first demo  (1 < n < N)
%    o time:  该变量可以通过两种方式提供。
%             如果所有演示之间的时间步长相同，则“时间”可以作为正标量给出(即time = dt)。
%             如果没有，'time'应该遵循以下格式:
%             - time{n}: 1 x T^n向量，表示第一次演示对应的长度为T^n的时间数组(1 < n < N)

%   o tol_cutting:  A small positive scalar that is used to trim data. It
%                   removes the redundant datapoint from the begining and
%                   the end of each demonstration that their first time
%                   derivative is less than 'tol_cutting'. Though this is
%                   not necessary for SEDS; however from practical point of
%                   view, it is very useful. There are always lots of noisy
%                   data at the begining (before the user starts the
%                   demosntration) and the end (after the user finished the
%                   demonstration) of each demosntration that are not
%                   useful.
%   tol_cutting:   一个小的正标量，用于裁剪数据。
%                  它从每个演示的开始和结束删除了冗余数据点，即它们的第一次导数小于“tol_cutting”。
%                  虽然这对于SEDS来说是不必要的;然而，从实际的角度来看，它是非常有用的。
%                  每次演示的开始(在用户开始演示之前)和结束(在用户完成演示之后)总是有大量无用的噪声数据。
%
% Outputs ----------------------------------------------------------------
%
%   o x0:      d x 1 array representing the mean of all demonstration's
%              initial points.
%   O x0:      d x 1数组表示所有演示初始点的平均值。
% 
%   o xT:      d x 1 array representing the mean of all demonstration's
%              final point (target point).
%   o xT:      d x 1数组表示所有演示终点的平均值。
%
%   o Data:    A 2d x N_Total matrix containing all demonstration data points.
%              Rows 1:d corresponds to trajectories and the rows d+1:2d
%              are their first time derivatives. Each column of Data stands
%              for a datapoint. All demonstrations are put next to each other 
%              along the second dimension. For example, if we have 3 demos
%              D1, D2, and D3, then the matrix Data is:
%                               Data = [[D1] [D2] [D3]]
%   o Data:    包含所有演示数据点的2d x N_Total矩阵。
%              行1:d对应于轨迹，行d+1:2d是它们的一阶导数。
%              Data的每一列代表一个数据点。
%              所有的演示都沿着第二次元挨个放置。
%              例如，如果我们有3个demo D1, D2和D3，那么矩阵Data为:
%                   Data = [[D1] [D2] [D3]]
%
%   o index:   A vector of N+1 components defining the initial index of each
%              demonstration. For example, index = [1 T1 T2 T3] indicates
%              that columns 1:T1-1 belongs to the first demonstration,
%              T1:T2-1 -> 2nd demonstration, and T2:T3-1 -> 3rd
%              demonstration.
%   o index:   由N+1个分量组成的向量，定义每个演示的初始索引。
%              例如index = [1 T1 T2 T3]表示列1:T1-1属于第一次演示，
%              列T1:T2-1 ->第二次演示，列T2:T3-1 ->第3次演示。
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Copyright (c) 2010 S. Mohammad Khansari-Zadeh, LASA Lab, EPFL,   %%%
%%%          CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The program is free for non-commercial academic use. Please contact the
% author if you are interested in using the software for commercial purposes.
% The software must not be modified or distributed without prior permission
% of the authors. Please acknowledge the authors in any academic publications
% that have made use of this code or part of it. Please use this BibTex
% reference:
% 
% S. M. Khansari-Zadeh and A. Billard, "Learning Stable Non-Linear Dynamical 
% Systems with Gaussian Mixture Models", IEEE Transaction on Robotics, 2011.
%
% To get latest upadate of the software please visit
%                          http://lasa.epfl.ch/khansari
%
% Please send your feedbacks or questions to:
%                           mohammad.khansari_at_epfl.ch

%%
%checking if a fixed time step is provided or not.
if length(time)==1
    dt = time;
end

d = size(demos{1},1); %dimensionality of demosntrations
Data=[];
index = 1;
for i=1:length(demos)
    clear tmp tmp_d
    
    
    % de-noising data (not necessary)
    for j=1:d
        tmp(j,:) = smooth(demos{i}(j,:),25); 
    end
    
    % computing the first time derivative
    if length(time)==1
        tmp_d = diff(tmp,1,2)/dt;
    else
        tmp_d = diff(tmp,1,2)./repmat(diff(time{i}),d,1);
    end
    
    % trimming demonstrations
    ind = find(sqrt(sum(tmp_d.*tmp_d,1))>tol_cutting);
    tmp = tmp(:,min(ind):max(ind)+1);
    tmp_d = tmp_d(:,min(ind):max(ind));
    
    % saving the initial point of each demo
    x0(:,i) = tmp(:,1);
    
    %saving the final point (target) of each demo
    xT(:,i) = demos{i}(:,end); 
    
    % shifting demos to the origin
    tmp = tmp - repmat(xT(:,i),1,size(tmp,2));
    
    % saving demos next to each other
    Data = [Data [tmp;tmp_d zeros(d,1)]];
    index = [index size(Data,2)+1];
end

xT = mean(xT,2); %we consider the mean value of all demonstraions' final point as the target
x0 = mean(x0,2); %the mean value of all demonstrations' initial point