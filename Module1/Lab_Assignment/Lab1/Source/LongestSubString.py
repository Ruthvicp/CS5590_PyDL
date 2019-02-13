"""
@author : ruthvicp
date : 2/11/2019
"""
# reference : https://www.geeksforgeeks.org/length-of-the-longest-substring-without-repeating-characters/
class LongestSubStr:
    def LongestSubStrWithLength(self, s):
        idx = 0
        max_l = 0
        longestSubStr = ''
        for position in range(1, len(s)):
            # print("position is ---> ", position)
            if(s[position] in s[idx:position]):
                # print("s is  ---> ", s[idx:position])
                max_l = len(s[idx:position]) if (len(s[idx:position]) > max_l) else max_l
                longestSubStr = s[idx:position]
                idx = s[idx:position].index(s[position]) + 1 + idx
                # print("idx is ---> ", idx)
            else:
                if(position == len(s) - 1):
                    max_l = max([max_l, len(s[idx:])])
                    # print("max is ---> ", max_l)
                    # print("s is -----> ", s[idx:])
                    if len(s[idx:]) > max_l :
                        longestSubStr = s[idx:]
        return longestSubStr, (max_l if(max_l != 0) else len(s))

input_str = input("Enter a string --> ")
if input_str is not None and len(input_str) > 0:
    # creating class object only if true
    s = LongestSubStr()
    output_str,len_output_str = s.LongestSubStrWithLength(input_str)
    print("Longest substring is {} and its length is {}".format(output_str,str(len_output_str)))
else :
    print("Enter a valid string")