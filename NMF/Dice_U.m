function [Dice,dice_mean,dice_max] = Dice_U(U)

[N,K] = size(U);
Dice = [];

for i = 1:K-1
    for j = i+1:K
        temp1 = U(:,i);
        temp2 = U(:,j);
        temp1(temp1~=0) = 1;
        temp2(temp2~=0) = 1;
        val = 2 * sum(temp1 .* temp2) /(sum(temp1) + sum(temp2));
        Dice = [Dice;val];
    end
end
dice_mean = mean(Dice);
dice_max = max(Dice);
        
        


