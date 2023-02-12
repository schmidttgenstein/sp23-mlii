# The main goal of this script is to demonstrate how the debugger works in VScode
# See an official tutorial video at https://code.visualstudio.com/docs/introvideos/debugging
# More details on this topic is available at https://code.visualstudio.com/docs/editor/debugging

print("Welcome to 741")

def addOne():
    x = 1
    print("Coding is fun!")
    print("The original value of x is %d" % x)
    x += 1
    return x

class fibSeq:
    def __init__(self, x): # The index in the Fibonacci sequence
        self.idx = x
        self.a = [1, 1]
    
    def eval(self, idx): 
        if idx == 1 or idx == 2:
            return self.a
        else:
            for i in range(idx-2):
                temp = self.a[0] + self.a[1]
                self.a[0] = self.a[1]
                self.a[1] = temp
            return self.a
    
    def next(self):
        return sum(self.eval(self.idx))

    def current(self):
        return self.eval(self.idx)[1]

    def prev(self):
        return self.eval(self.idx)[0]

    def __str__(self):
        return f'The number at index {self.idx} in a Fibonacci sequence is {self.current()}'

sqr = addOne()**2 # Compute the square the return value of addOne
seq = fibSeq(7) # Get the number at index 7 in a Fibonacci sequence

print(seq)
print("The current value of x is ",addOne(), sep='')
print("The square of x is ",sqr, sep='')


