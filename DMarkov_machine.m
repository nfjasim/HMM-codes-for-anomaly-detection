function P = DMarkov_machine(X, NS, D)
% X is a symbol sequence
% NS = Alphabet size
% D = Markov depth
% P = State stationary probability vector
A=[1:NS];
if D==1
SV=[1:NS]';
else
C = cell(D, 1);             %Preallocate a cell array
[C{:}] = ndgrid(A);         %Create L grids of values
S = cellfun(@(A){A(:)}, C); %Convert grids to column vectors
SV = [S{:}];  % State vector
end
T=length(X);
    MM=zeros(NS^D,NS^D); %%% State transition Matrix
for i=1:NS^D
    k1=0; 
    for ti=1:T-D+1
        if X(ti:ti+D-1)==SV(i,:)
            k1=k1+1;
        end
    end
    for j=1:NS^D
        if D==1 || isequal(SV(i,2:end),SV(j,1:end-1))
        rr=[SV(i,:) SV(j,end)];
        k2=0;
        for ti=1:T-D
            if X(ti:ti+D)==rr
                k2=k2+1;
            end
        end
        if k1>0
        MM(i,j)=k2/k1;
        end
        end
    end
end
P=(inv((MM'-eye(NS^D)+ones(NS^D,1)*ones(1,NS^D)))*ones(NS^D,1))';