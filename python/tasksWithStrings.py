def allCharsUnique(str): 
    return len(set(str)) == len(str)  


def isPermutation(str1, str2):
    str1 = str1.replace(' ', '')
    str2 = str2.replace(' ', '')
    if((len(str1) == 0) | (len(str2) == 0)):
        return False
    print(str1)
    print(str2)
    dict_str1 = {}
    for ch in str1: 
        if(dict_str1.get(ch) == None):
            dict_str1.update({ch:1})
        else:
            dict_str1.update({ch:dict_str1.get(ch) + 1})
    print(dict_str1)
    for ch in str2:
        if(dict_str1.get(ch) == None):
            return False
        counter = dict_str1.get(ch)
        if(counter == 1):
            dict_str1.pop(ch)
        else:
            dict_str1.update({ch:counter - 1})

    return len(dict_str1) == 0

def isPermutation_str(str1, str2): 
    str1 = str1.replace(' ', '')    
    str2 = str2.replace(' ', '')

    if((len(str1) != len(str2)) | (len(str1) == 0)):
        return False

    for ch in str2:
        if ch in str1:
            str1 = str1.replace(ch, '')
            
    return str1 == ''

    
#print(allCharsUnique('unique'))
#print(isPermutation_str('   ', ''))
#print(isPermutation_str('ab', 'ba'))
print(isPermutation_str('the cow is on the moon','the moon is on the cow'))