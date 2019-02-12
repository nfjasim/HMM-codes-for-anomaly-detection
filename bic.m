function [numComponents] = bic(data, max_num_mix_comp)
BIC=zeros(1,max_num_mix_comp);
gmmodels=cell(1,max_num_mix_comp);
options=statset('MaxIter',500);
for k=1:max_num_mix_comp
 %   gmmodels{k}=fitgmdist(data',k,'Options',options,'CovarianceType','diagonal', 'RegularizationValue', 0.1);
    gmmodels{k}=gmdistribution.fit(data',k,'Options',options,'covtype','diagonal', 'Regularize', 0.1);
    BIC(k)=gmmodels{k}.AIC;
end
[minAIC,numComponents]=min(BIC);
