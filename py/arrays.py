class ArrayProbs:
  def contains_duplicate(self, nums: list[int]) -> bool:
    # nums.sort()
    # prev_num = None

    # for i,num in enumerate(nums):
    #   if prev_num == num:
    #     return True
    #   else:
    #     prev_num = num
    
    # return False

    it = {}

    for i, num in enumerate(nums):
      if num in it:
        return True
      else:
        it[num] = 1
      
    return False




  def valid_anagram(self, s: str, t: str) -> bool:
    s.sort()
    t.sort()

    for i,x in enumerate(s):
      if s[i] != t[i]:
        return False
    
    return True
  
  def twoSum(self, nums: list[int], target: int) -> list[int]:
    
    hash = {}

    for i, num in enumerate(nums):
      hash[num] = i

    for k,v in hash.items():
      if target - k in hash:
        answer = [v, hash[target - k]]
        answer.sort()
        return answer
    
    return []
