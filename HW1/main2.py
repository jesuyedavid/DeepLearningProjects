import numpy as np
#import scipy.misc 


def main():
    x=np.random.random((1,784))
    w=np.random.random((784,1))
    print(myNeuralNet(x,w))

def myNeuralNet(x,w):    
    mulxy=np.dot(x,w)[0][0]
    b=2#bias
    y=mulxy+b
    return(y)

if __name__ == "__main__":
    main()




