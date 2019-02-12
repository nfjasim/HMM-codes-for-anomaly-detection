function X = mep_symbolization(y, xpmin)
% code written by Najah Ghalyan
NP=size(xpmin,2);
X(y < xpmin(1)) = 1;    
    for p=1:NP  
        X(y>=xpmin(p))=p+1;  
    end
end
