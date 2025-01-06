function [H, loss]= recon_H(X_test, W)

num_iter = 100;
[~,K] = size(W);
N = size(X_test,2);

H = rand(K,N);
for i = 1:num_iter
    H = H .* ((W'*X_test) ./ (W'*W*H + 1e-12));
    loss(i) = norm(X_test-W*H,'fro')/norm(X_test,'fro');
    fprintf('Update-Error: %d\n',loss(i));
end
H = H';
