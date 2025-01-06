function retest_value = retest(U1,U2)


R1 = size(U1,2);
R2 = size(U2,2);

similarity = zeros(R1,R2);

for i = 1:R1
    for j  = 1:R2
        similarity(i,j) = corr(U1(:,i),U2(:,j));
    end
end

similarity = abs(similarity);
temp = similarity;
temp(find(temp<0)) = 0.001;
cost_matrix = 1 ./ temp;
[index1, index2] = graph_matching_min(cost_matrix);

best_match = zeros(1,R1); 

for i = 1:R1
    best_match(i) = similarity(index1(i), index2(i));
end
retest_value = mean(best_match);
