

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        num3 =sorted(nums1+nums2)
        slong =int(len(num3)/2)
        if len(num3)%2==0:
            return (num3[slong]+num3[slong-1])/2
        else:
            return (num3[slong])

nums1 = [1,3]
nums2 = [2,4]

print(Solution.findMedianSortedArrays(Solution,nums1,nums2))



