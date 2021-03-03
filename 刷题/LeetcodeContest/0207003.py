def largestMerge(word1: str, word2: str):
    def comprasion(word1, word2):
        for i in range(min(len(word1), len(word2))):
            if word1[i] < word2[i]:
                # merge += word2[0]
                s = word2[0]
                word2 = word2[1:]
                # return word2[0]
                return s
            elif word1[i] > word2[i]:
                # merge += word1[0]
                s = word1[0]
                word1 = word1[1:]
                return s
            else:
                i += 1
            s = word1[0]
            word1 = word1[1:]
            return s
    # ---
    merge = ""
    while word1 and word2:
        # i = 0
        print("2222222222222")
        merge += comprasion(word1, word2)
    if word1:
        merge += word1
        print(merge)
        # return merge
    elif word2:
        merge += word2
        print(merge)
        return merge

if __name__ == "__main__":
    print("1111111111")
    largestMerge("cabaa", "bcaaa")
