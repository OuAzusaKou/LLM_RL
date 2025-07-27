nums = [0,1,2,2,1,0]
max_sum = 0
for i in range(len(nums)-60):
    sum1 = sum(nums[i:i+60])
    for j in range(i+60, len(nums)-60):
        max_sum2 = 0 
        sum2 = sum(nums[j:j+60])
        if sum2 > max_sum2:
            max_sum2 = sum2
print(max_sum)