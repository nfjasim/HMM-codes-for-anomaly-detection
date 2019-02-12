function [M, mu1, Sigma1, mixmat1, prior1, transmat1] = estimated_hmm(data, Q, max_num_mix_comp)
%  M = aic(data, max_num_mix_comp)
 M=1;
 %M = bic(data, max_num_mix_comp);
 trial=1;
 [prior0, transmat0, mixmat0, mu0, Sigma0] =  init_mhmm(data, Q, M, 'diag');
 max_iter = 100; 
 while trial<100
try
[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    learn_mhmm(data, prior0, transmat0, mu0, Sigma0, mixmat0,100);

if ~iscell(data)
     data_cache = num2cell(data, [1 2]);
end
     O_cache = size(data_cache{1},1);
     Q_cache = length(prior1);
     M_cache= size(mixmat1,2);
     
      for j=1:Q_cache
        for k=1:M
            if rank(Sigma1(:,:,j,k))==O_cache; 
                break
            end
        end
        break
      end
     clear O_cache;clear Q_cache; clear M_cache; clear data_cache;
break
catch
    [prior0, transmat0, mixmat0, mu0, Sigma0] =  ...
         init_mhmm(data, Q, M, 'spherical');
     
     end
trial
trial=trial+1;
 end       

