
# SlidingWindow Algorithm
def maxSlidingWindow(nums, k):
    left, right = 0, k - 1
    n = len(nums)
    result = []
    while right < n:
        window = nums[left:right+1]
        result.append(max(window))
        left += 1
        right += 1
    return result

# Run
nums = [1,3,-1,-3,5,3,6,7]
k = 2
print(maxSlidingWindow(nums, k))
