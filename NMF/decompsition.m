function [U_nor,U,INFO] = decompsition(X, options)
%
% [W,H,INFO] = NMFL0_W(X, options)
%
% run NMFL0_W described in 
% R. Peharz and F. Pernkopf, "Sparse nonnegative matrix factorization with
% â„?0-constraints", Neurocomputing, 2012.
%
% The algorithm returns an approximate solution for 
%
%   minimize  ||X - W * H||_2
%   s.t.      W(:) >= 0
%             H(:) >= 0
%             sum(W(:,k) > 0) <= L   for all k
%
% w.r.t. W and H
%
%
% input:   
%  X nonnegative data matrix 
%  options: structure of parameters:
%    K:                 number of columns in dictionary matrix W
%    L:                 maximal number of nonzeros in each column of W
%    numIter:           number of (outer) iterations
%    update type:       - 'MU'multiplicative updates according to Lee and Seung, 
%                       "Algorithms for nonnegative matrix factorization", 2001.
%                 
%    numUpdateIter:     number of update (innner) iterations.
%    VERBOSITY:         verbose mode if not 0; default 1.
%    H:                 initial H; if empty, random numbers are used for
%                       initialization.
%
% output:
%  W:       dictionary matrix
%  H:       coding matrix
%  INFO:    structure of some info
%     E:      error in each iteration ||X - W*H||_F)
%     UDtime: time needed by dictionary updates
%
% Robert Peharz, 2011
%

if any(X(:)<0),                                  error('X contains negative values.'); end
if ~isfield(options,'K'),                        error('options must contain parameter K (number of bases).'); end
if ~isfield(options,'L'),                        error('options must contain parameter L (maximal number of bases per signal).'); end
if ~isfield(options,'numIter'),                  error('options must contain parameter numIter.'); end
if ~isfield(options,'numUpdateIter'),            error('options must contain parameter numUpdateIter.'); end


if isfield(options,'verbosity')
    VERBOSITY = options.verbosity;
else
    VERBOSITY = 1;
end

[D,N] = size(X);
K =             options.K;
L =             options.L;
numIter =       options.numIter;
numUpdateIter = options.numUpdateIter;

if nargout > 2 
    E = zeros(numIter,1);    
    UDtime = zeros(numIter,1);
    INFO = [];
end

if isfield(options,'H')
    if any(options.H(:)<0), error('options.H contains negative values.'); end
    H = options.H;
else
    H = rand(K,N);
end

W_mat = cell(numIter,1);
H_mat = cell(numIter,1);
for iter=1:numIter
    if VERBOSITY
        fprintf('Iteration: %d   ',iter);
    end
    
    W = sparseNNLS(X',H',[],[],K,K);
    W = W';
       
    % Set K-L smallest values to zero for each atom
    for p=1:K
        [~, idx] = sort(W(:,p),'ascend');
        W(idx(1:D-L),p) = 0;
    end
    
    %%% Update Stage

    tic
          for k=1:numUpdateIter
                H = H .* ((W'*X) ./ (W'*W*H + 1e-12));
                if k < numUpdateIter
                    W = W .* ((X*H') ./ (W*H*H' + 1e-12));
                end
            end
      elapsedT = toc;
     W_mat{iter,1} = W;
     H_mat{iter,1} = H;

    if nargout > 2
        E(iter) = norm(X-W*H,'fro');
        UDtime(iter) = elapsedT;
        loss(iter) = norm(X-W*H,'fro')/norm(X,'fro');
        fprintf('Update-Error: %d\n',norm(X-W*H,'fro')/norm(X,'fro'));
    end
end

[min_loss, min_index] = min(loss);
W = W_mat{min_index,1};
H = H_mat{min_index,1};
final_loss = norm(X-W*H,'fro')/norm(X,'fro');

%normW = sqrt(sum(W.^2));
%H = diag(normW) * H;
%W = W * diag(1./normW);
U{1} = W;
U{2} = H';

U_max = max(abs(W));
U_nor{1} = W * diag(1./U_max);
U_nor{2} = H' * diag(U_max);



if nargout > 2
    INFO.E = E;
    INFO.UDtime = UDtime;
    INFO.loss = loss;
    INFO.final_loss = final_loss;
end


