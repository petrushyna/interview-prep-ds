def detectRequiredSum(list_num, sum):
    if(len(list_num) != len(set(list_num))):
        return []
    if((list_num == []) | sum == None):
        return []
    dict_list = {}
    for ind, num in enumerate(list_num):
        dict_list.update({num:ind})
        print(dict_list)

    list_num.sort()
    list_num2 = list_num.copy()

    for ind1, num in enumerate(list_num): 
        p = round(len(list_num)/2)
        check_sum = 2*list_num[p-1]
        if(check_sum > sum):
            list_num = list_num[:p]
            print(list_num)
        elif(check_sum < sum):
            list_num = list_num[p:]
            print(list_num)
        else:
            return [p-1, p-1]
        num = list_num[0]
        ind1 = 0
        for num2 in list_num2:
            if((num + num2) == sum):
                print(num)
                print(num2)
                return([dict_list.get(num), dict_list.get(num2)])

def detectRequiredSum_better(list_num, sum):
    if(len(list_num) != len(set(list_num))):
        return []
    if((list_num == []) | sum == None):
        return []
    dict_list = {}
    for ind1, num1 in enumerate(list_num):
        if(dict_list.get(num1) != None): 
            return[dict_list.get(num1), ind1]
        required_element = sum - num1
        print(required_element)
        dict_list.update({required_element:ind1})
        print(dict_list)
    return[]
            

            
print(detectRequiredSum_better([1,3,7,2,11], 9))
                