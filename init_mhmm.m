function [init_state_prob, transmat, mixmat, mu, Sigma] =  init_mhmm(data, Q, M, cov_type)
% INIT_MHMM Compute initial param. estimates for an HMM with mixture of Gaussian outputs.
% [init_state_prob, transmat, mixmat, mu, Sigma] = init_mhmm(data, Q, M, cov_type)
%
% INPUTS:
% data{ex}(:,t) or data(:,t,ex) if all sequences have the same length
% Q = num. hidden states
% M = num. mixture components
% cov_type = 'full', 'diag' or 'spherical'
%
% OUTPUTS:
% init_state_prob(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% mixmat(j,k) = Pr(M(t)=k | Q(t)=j) where M(t) is the mixture component at time t
% mu(:,j,k) = mean of Y(t) given Q(t)=j, M(t)=k
% Sigma(:,:,j,k) = cov. of Y(t) given Q(t)=j, M(t)=k
%
% init_state_prob, transmat and mixmat are initialized randomly.
% mu and Sigma are initialized using K-means.

if iscell(data)
  data = cat(2, data{:});
  O = size(data, 1);
else
  [O T nex] = size(data);
  data = reshape(data, [O T*nex]);
end

init_state_prob = normalise(ones(Q,1)/Q);
transmat = mk_stochastic(rand(Q));

% init_state_prob = normalise(rand(Q,1));
% transmat = mk_leftright_transmat(Q,0.5);



if M > 1
  rand('state',sum(100*clock));
  mixmat = mk_stochastic(rand(Q,M));
else
  mixmat = ones(Q, 1);
end

[mu, Sigma] = mixgauss_init(Q*M, data, cov_type);
mu = reshape(mu, [O Q M]);
Sigma = reshape(Sigma, [O O Q M]);