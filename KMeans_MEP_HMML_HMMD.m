%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Code written by Najah Ghalyan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% K-means
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clc
clear
Y=importdata('combustion_data.mat');
NSP=size(Y,1);
if iscell(Y)
thr=cell2mat(Y(:,2));
end
window_length =50;
thr = round(thr./window_length);
alphabet_size = 2; Q = alphabet_size;
depth = 2; D = depth; net_length=10000;
    ii=0;
 for kt=1:NSP
     T = window_length;
     if iscell(Y)
     YY=cell2mat(Y(kt,1));
     end
     TT=size(YY,2);
     interval=floor(TT/net_length);
     YY=downsample(YY,interval);
     thr(kt)=thr(kt)/interval;
     Nt =floor(size(YY,2)/window_length);
    for i=1:Nt
        ii=ii+1;
        if i<=thr(kt)
            class(ii)=1;
        else
            class(ii)=2;
        end
    end
 for t=1:Nt
     data=YY((t-1)*T+1:t*T);
     if t==1
         [idx, centroids] = kmeans(data',Q);
     end
     %%%%%%%%%%%%%%%%%% K-means Symbolization %%%%%%%%%%%%%%%%
     X = knnsearch(centroids,data')';
     P = DMarkov_machine(X, Q, depth);
     Pst(t,:)=P;
 end
  %%%%%% KL-divergence 
clearvars dpq pq 
  for k=1:Nt
    for j=1:Q^D
        if Pst(k,j)==0
            pq(j)=0;
        else
             pq(j)=Pst(k,j)*log2(Pst(k,j)/Pst(1,j));
        end
    end
    dpq(k)=sum(pq);
end
dpq2=dpq/max(dpq);
if kt==1
    d=dpq2;
else
    d=[d dpq2];
end
 kt
 end
 n1=max(size(class(class==1))); n2=max(size(class(class==2)));
[yd,I]=sort(d,'descend');
auc=0;
pd(1)=0;
fp(1)=0;
inc=1;
for i=1:max(size(d))
    if class(I(i))==2
        pd(inc)=pd(inc)+1/n2;
    else
        auc=auc+(1/n1)*pd(inc);
        inc=inc+1;
        fp(inc)=fp(inc-1)+1/n1;
        pd(inc)=pd(inc-1);
    end
end
auc_kmeans=auc;
hold on
plot(fp,pd, 'b--')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% MEP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except auc_kmeans Y NSP net_length alphabet_size depth window_length class
if iscell(Y)
thr=cell2mat(Y(:,2));
end
thr = round(thr./window_length);
Q = alphabet_size;
D = depth; 
    ii=0;
 for kt=1:NSP
     T = window_length;
     if iscell(Y)
     YY=cell2mat(Y(kt,1));
     end
     TT=size(YY,2);
     interval=floor(TT/net_length);
     YY=downsample(YY,interval);
     thr(kt)=thr(kt)/interval;
     Nt =floor(size(YY,2)/window_length);
 for t=1:Nt
     data=YY((t-1)*T+1:t*T);
     if t==1
         xpmin = mep_partition(data, Q);
     end
     %%%%%%%%%%%%%%%%%% MEP Symbolization %%%%%%%%%%%%%%%%
     X = mep_symbolization(data, xpmin);
     P = DMarkov_machine(X, Q, depth);
     Pst(t,:)=P;
 end
  %%%%%% KL-divergence 
clearvars dpq pq 
  for k=1:Nt
    for j=1:Q^D
        if Pst(k,j)==0
            pq(j)=0;
        else
             pq(j)=Pst(k,j)*log2(Pst(k,j)/Pst(1,j));
        end
    end
    dpq(k)=sum(pq);
end
dpq2=dpq/max(dpq);
if kt==1
    d=dpq2;
else
    d=[d dpq2];
end
 kt
 end
 n1=max(size(class(class==1))); n2=max(size(class(class==2)));
[yd,I]=sort(d,'descend');
auc=0;
pd(1)=0;
fp(1)=0;
inc=1;
for i=1:max(size(d))
    if class(I(i))==2
        pd(inc)=pd(inc)+1/n2;
    else
        auc=auc+(1/n1)*pd(inc);
        inc=inc+1;
        fp(inc)=fp(inc-1)+1/n1;
        pd(inc)=pd(inc-1);
    end
end
auc_mep=auc;
hold on
plot(fp,pd, 'k --')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% HMM-L
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except auc_kmeans auc_mep Y NSP net_length alphabet_size depth window_length class
if iscell(Y)
thr=cell2mat(Y(:,2));
end
thr = round(thr./window_length);
hmm_numstates=alphabet_size;
Q = hmm_numstates;
D = depth;
max_num_mix_comp = 2; 
NY = min(size(Y)); 
    ii=0;
 for kt=1:NSP
     T = window_length;
     if iscell(Y)
        YY=cell2mat(Y(kt,1));
     end
     TT=size(YY,2);
     interval=floor(TT/net_length);
     YY=downsample(YY,interval);
     thr(kt)=thr(kt)/interval;
     Nt =floor(size(YY,2)/window_length);
 for t=1:Nt

     data=YY((t-1)*T+1:t*T);
     if t==1
     [M, mu1, Sigma1, mixmat1, prior1, transmat1] = estimated_hmm(data, hmm_numstates, max_num_mix_comp);
     end
     [B, B2] = mk_mhmm_obs_lik(data, mu1, Sigma1, mixmat1);
     [loglik, alpha] = hmmFilter(prior1, transmat1, B); 
     LL(t)=loglik;
 end
clearvars dpq 
for k=1:Nt
    dpq(k)=LL(1)-LL(k);
end
dpq2=dpq/max(dpq);
if kt==1
    d=dpq2;
else
    d=[d dpq2];
end
 kt
 end
 n1=max(size(class(class==1))); n2=max(size(class(class==2)));
[yd,I]=sort(d,'descend');
auc=0;
pd(1)=0;
fp(1)=0;
inc=1;
for i=1:max(size(d))
    if class(I(i))==2
        pd(inc)=pd(inc)+1/n2;
    else
        auc=auc+(1/n1)*pd(inc);
        inc=inc+1;
        fp(inc)=fp(inc-1)+1/n1;
        pd(inc)=pd(inc-1);
    end
end
auc_hmmL=auc;
hold on
box on
plot(fp,pd, 'k')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% HMM-D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except auc_hmmL auc_kmeans auc_mep Y NSP net_length alphabet_size depth window_length class
if iscell(Y)
thr=cell2mat(Y(:,2));
end
thr = round(thr./window_length);
hmm_numstates=alphabet_size;
Q = hmm_numstates;
D = depth;
max_num_mix_comp = 2; 
NY = min(size(Y)); 
    ii=0;   
     for kt=1:NSP
     T = window_length;
     if iscell(Y)
     YY=cell2mat(Y(kt,1));
     end
     TT=size(YY,2);
     interval=floor(TT/net_length);
     YY=downsample(YY,interval);
     thr(kt)=thr(kt)/interval;
     Nt =floor(size(YY,2)/window_length);
    for i=1:Nt
        ii=ii+1;
        if i<=thr(kt)
            class(ii)=1;
        else
            class(ii)=2;
        end
    end
 for t=1:Nt
%      if t==Nt
%          T=length(YY)-(Nt-1)*T;
%      end
     data=YY((t-1)*T+1:t*T);
     if t==1
     [M, mu1, Sigma1, mixmat1, prior1, transmat1] = estimated_hmm(data, hmm_numstates, max_num_mix_comp);
     end
     %%%%%%%%%%%%%%%%%% Viterbi %%%%%%%%%%%%%%%%
     %%%% Find the most-probable (Viterbi) path {q(t)}
     B = mk_mhmm_obs_lik(data, mu1, Sigma1, mixmat1);
     X = viterbi_path(prior1, transmat1, B);
     P = DMarkov_machine(X, Q, depth);
     Pst(t,:)=P;
 end
  %%%%%% KL-divergence 
clearvars dpq pq 
  for k=1:Nt
    for j=1:Q^D
        if Pst(k,j)==0
            pq(j)=0;
        else
             pq(j)=Pst(k,j)*log2(Pst(k,j)/Pst(1,j));
        end
    end
    dpq(k)=sum(pq);
end
dpq2=dpq/max(dpq);
if kt==1
    d=dpq2;
else
    d=[d dpq2];
end
 kt
 end
 n1=max(size(class(class==1))); n2=max(size(class(class==2)));
[yd,I]=sort(d,'descend');
auc=0;
pd(1)=0;
fp(1)=0;
inc=1;
for i=1:max(size(d))
    if class(I(i))==2
        pd(inc)=pd(inc)+1/n2;
    else
        auc=auc+(1/n1)*pd(inc);
        inc=inc+1;
        fp(inc)=fp(inc-1)+1/n1;
        pd(inc)=pd(inc-1);
    end
end
auc_hmmD=auc;
hold on
box on
plot(fp,pd, 'r')
xlim([0 1]); ylim([0 1]);
xlabel('Probability of False Alarm'); ylabel('Probability of True Detection');
title(['AUC via: K-means = ' num2str(auc_kmeans) ', MEP = ' num2str(auc_mep) ', HMM-D = ', num2str(auc_hmmD) ', HMM-L = ', num2str(auc_hmmL)])
legend('Kmeans','MEP', 'HMM-L', 'HMM-D')