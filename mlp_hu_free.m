function net = mlp_hu_free(TrainingData)
nin=size(TrainingData,1)/2;
Inputs=TrainingData(1:nin,:)';
TargetData=TrainingData(nin+1:end,:)';
L = 200;
varepsi = 1e-3;
N = 0;
X = Inputs;
Y = TargetData;
error = Y;
w = [];
beta0 = [];
actFun = @(tempH) 2 ./ (1 + exp(-2*tempH))-1;
while N < L && norm(error) > varepsi
    Wnew = rand(size(Inputs, 2), 1);
    w = [w, Wnew];
    tempH = X*Wnew;
    H = actFun(tempH);
    clear tempH;
    beta = H'*error./(H'*H);
    beta0 = [beta0; beta];
    adjust=max(abs(beta0),[],2);
    beta0=abs(beta0./adjust);
    error = error - H*beta;
    N = N+1;
end 
b = zeros(L, 2);
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'off');
beta_opt = fmincon(@(beta0) norm(activation(w, X) * beta0 - Y, 'fro')^2, ...
                beta0, [], [], [], [], [], [], ...
                @(beta0) constraints(w, beta0, b), options);
ielm.outputWeight = beta_opt;
ielm.inputWeight = w;
net = ielm;
save('NetParaOpt.mat', "net", 'beta0')
end
function phi = activation(w, x)
    phi = 2 ./ (1 + exp(-2 * (x * w))) - 1;
end
function [c, ceq] = constraints(beta, w, b)
    c = beta' .* w; 
    ceq = b; 
end




