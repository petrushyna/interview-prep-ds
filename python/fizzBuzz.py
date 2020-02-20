def fizzBuss(list, num_to_print):
    
    for ind in range(0,len(list)):
        if((list[ind] % 15) == 0):
            list[ind] = 'FizzBuzz'
        elif((list[ind] % 5) == 0):
            list[ind] = 'Buzz'
        elif((list[ind] % 3) == 0):
            list[ind] = 'Fizz'
    return list[:num_to_print]

print(fizzBuss([1,5,4,3,15,45,6],7))