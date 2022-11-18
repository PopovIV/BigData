import math
n = int(input())

num1 = int(n)
num2 = int(n)
i = 2
while i < int(math.sqrt(n)) + 1:
    if num1 % i == 0:
        num1 += 1
        i = 1
    i += 1
i = 2
while i < int(math.sqrt(n)) + 1:
    if num2 % i == 0:
        num2 -= 1
        i = 1
    i += 1
print(num2, num1)