function [y] = mlpfwd(net, x)
% Inputs -----------------------------------------------------------------
%   o net:      network discription
%   o x:       d x N matrix vector representing N different starting point(s)

tempH=x*net.inputWeight; %+net.biasOfHiddenNeurons;
H = 2 ./ (1 + exp(-2*tempH))-1;
% switch net.activefn
%     case {'sig','sigmoid'}
%         %%%%%%%% Sigmoid 
%         H = 2 ./ (1 + exp(-tempH))-1;
%     case {'sin','sine'}
%         %%%%%%%% Sine
%         H = sin(tempH);  
%     case {'tanh'}
%         H=tanh(tempH);
%     case {'sinh'}
%         H=sinh(tempH);
%     case {'hardlim'}
%         %%%%%%%% Hard Limit
%         H = hardlim(tempH); 
%         %%%%%%%% More activation functions can be added here 
%     otherwise
%         error('wrong active function!');
% end
% 
% switch net.outfn

%   case 'linear'    % Linear outputs

    y = net.outputWeight'*H';
    
end
