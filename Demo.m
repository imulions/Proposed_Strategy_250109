close all
clear
clc
load PathS1to4.mat
load DisData.mat
load DirPath.mat
dt = 0.001;
tol_cutting = 0;
options.energy='on';
options.suf='off';
demos = {};
demos{1,1} = pathS(:,1:2)';
demos{1,2} = pathS1(:,1:2)';
demos{1,3} = pathS2(:,1:2)';
demos{1,4} = pathS3(:,1:2)';
demos{1,5} = pathS4(:,1:2)';
[x0, xT, Data, index] = preprocess_demos(demos,dt,tol_cutting);
options.max_iter=1000;
options.alpha=0;
options.tol_stopping=10^-10;         
options.display = 1;                                     
options.alpha=0.05;         
start_time_train=cputime;
nhidden=150;
% net=mlp_hu_free(Data);
load("NetParaOpt.mat")
end_time_train=cputime;
end_time_train=end_time_train-start_time_train; 
d = size(Data, 1)/2; 
opt_sim.dt = 0.001;
opt_sim.i_max = 100000;
opt_sim.tol = 0;
x0_all = x_exp(1:2, 1) + [2; 8];
fn_handle = @(x) mlpfwd(net, x'); 
[x xd t] = SimulationUnderDisR1(x0_all,[],fn_handle,v_exp,x_exp,DisData,opt_sim);
x_all=[];
for i=1:size(x,3)
    x_all=[x_all x(:,:,i)];
end
e = [];
for i = 1 : size(x, 3)
    e(:, :, i) = abs(x(:,:,i) - x_exp(:, :, i));
end
figure(1)
hold on
plot(x_exp(1,:),x_exp(2,:),'r.','linewidth',1.5);
n=size(x_all,2)/size(x,3);
for i=1:size(x,3)
    plot(x_all(1,n*(i-1)+1:n*i),x_all(2,n*(i-1)+1:n*i),'k','linewidth',2)
end
set(gca,'xtick',[],'xticklabel',[]);
set(gca,'ytick',[],'yticklabel',[])
