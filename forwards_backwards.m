function [alpha, beta, gamma, xi, loglik, gamma2] = forwards_backwards(...
    prior, transmat, obslik, obslik2, mixmat, act, filter_only)
% FORWARDS_BACKWARDS Compute the posterior probs. in an HMM using the forwards backwards algo.
%
% Use [ALPHA, BETA, GAMMA, XI, LOGLIK] = FORWARDS_BACKWARDS(PRIOR, TRANSMAT, OBSLIK)
% for HMMs where the Y(t) depends only on Q(t).
% Use OBSLIK = MK_DHMM_OBS_LIK(DATA, B) or OBSLIK = MK_GHMM_OBS_LIK(DATA, MU, SIGMA) first.
% If the sequence is of length 1, XI will have size S*S*0, and ALPHA=GAMMA and BETA = 1.
%
% Use [ALPHA, BETA, GAMMA, XI, LOGLIK, GAMMA2] = FORWARDS_BACKWARDS(PRIOR, TRANSMAT, OBSLIK, OBSLIK2, MIXMAT)
% for HMMs where Y(t) depends on Q(t) and M(t), the mixture component.
% Use [OBSLIK, OBSLIK2] = MK_MHMM_OBS_LIK(DATA, MU, SIGMA, MIXMAT) first.
% 
% Use [ALPHA, BETA, GAMMA, XI, LOGLIK, GAMMA2] = FORWARDS_BACKWARDS(PRIOR, TRANSMAT, OBSLIK, [], [], ACT)
% for POMDPs, where ACT(t) is the action that inputs to Q(t) (so act(1) is ignored)
%
% FORWARDS_BACKWARDS(PRIOR, TRANSMAT, OBSLIK, [], [], ACT, FILTER_ONLY) with FILTER_ONLY = 1
% will just run the forwards algorithm. In this case, alpha = gamma, and beta = gamma2 = [].
% 
% Inputs:
% PRIOR(I) = Pr(Q(1) = I)
% TRANSMAT(I,J) = Pr(Q(T+1)=J | Q(T)=I)
% OBSLIK(I,T) = Pr(Y(T) | Q(T)=I)
%
% For mixture models only:
% OBSLIK2(I,K,T) = Pr(Y(T) | Q(T)=I, M(T)=K)
% MIXMAT(I,K) = Pr(M(T)=K | Q(T)=I)
%
% For POMDPs, transmat{a}(i,j) = Pr(Q(t+1)=j | Q(t)=i, A(t)=a)
%
% Outputs:
% alpha(i,t) = Pr(X(t)=i, O(1:t))
% beta(i,t)  = Pr(O(t+1:T) | X(t)=i)
% gamma(i,t) = Pr(X(t)=i | O(1:T))
% xi(i,j,t)  = Pr(X(t)=i, X(t+1)=j | O(1:T)) t <= T-1
% gamma2(j,k,t) = Pr(Q(t)=j, M(t)=k | O(1:T))
%

if ~exist('obslik2') | isempty(obslik2)
  mix = 0;
  M = 1;
else
  mix = 1;
  M = size(mixmat, 2);
end

T = size(obslik, 2);

if ~exist('act')
  act = ones(1, T);
  transmat = { transmat };
end

if ~exist('filter_only')
  filter_only = 0;
end

Q = length(prior);

scale = ones(1,T);
% scale(t) = Pr(O(t) | O(1:t-1))
% Hence prod_t scale(t) = Pr(O(1)) Pr(O(2)|O(1)) Pr(O(3) | O(1:2)) ... = Pr(O(1), ... ,O(T)) = P
% or log P = sum_t log scale(t)
%
% Note, scale(t) = 1/c(t) as defined in Rabiner
% Hence we divide beta(t) by scale(t).
% Alternatively, we can just normalise beta(t) at each step.

loglik = 0;
prior = prior(:); 

alpha = zeros(Q,T);
beta = zeros(Q,T);
gamma = zeros(Q,T);
xi = zeros(Q,Q,T-1);
gamma2 = zeros(Q,M,T);


t = 1;
alpha(:,1) = prior .* obslik(:,t);
[alpha(:,t), scale(t)] = normalise(alpha(:,t));
for t=2:T
  alpha(:,t) = (transmat{act(t)}' * alpha(:,t-1)) .* obslik(:,t);
  [alpha(:,t), scale(t)] = normalise(alpha(:,t));
  xi(:,:,t-1) = normalise((alpha(:,t-1) * obslik(:,t)') .* transmat{act(t)});
end

if filter_only
  beta = [];
  gamma = alpha;
  gamma2 = [];
else
  beta(:,T) = ones(Q,1);
  gamma(:,T) = normalise(alpha(:,T) .* beta(:,T));
  t=T;
  if mix
    denom = obslik(:,t) + (obslik(:,t)==0); % replace 0s with 1s before dividing
    gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M]);
  end
  for t=T-1:-1:1
    b = beta(:,t+1) .* obslik(:,t+1); 
    %beta(:,t) = (transmat{act(t)} * b) / scale(t);
    beta(:,t) = normalise((transmat{act(t+1)} * b));
    gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));
    xi(:,:,t) = normalise((transmat{act(t+1)} .* (alpha(:,t) * b')));
    if mix
      denom = obslik(:,t) + (obslik(:,t)==0); % replace 0s with 1s before dividing
      gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M]);
    end
  end
end
loglik = sum(log(scale));