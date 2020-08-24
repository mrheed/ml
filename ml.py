import math, random

class Matrix:
    def __init__(self, data):
        self.matrix = data
        self.row = len(data)
        self.col = len(data[0])

    def add(self, matrix):
        for i in range(self.row):
            for j in range(self.col):
                if len(matrix) == 1:
                    self.matrix[i][j] += matrix[0]
                else:
                    self.matrix[i][j] = matrix[i][j]
        return Matrix(self.matrix)
    
    def dot(self, matrix):
        m1_r = self.row
        m1_c = self.col
        m2_r = len(matrix)
        m2_c = len(matrix[0])
        if m1_c != m2_r: 
            raise("error: matricies dimension does not correct")
        def s_3(i, j): 
            return sum([self.matrix[i][k] * matrix[k][j] for k in range(m1_c)])
        def s_2(i): 
            return [s_3(i, j) for j in range(m2_c)]
        return Matrix([s_2(i) for i in range(m1_r)])

    def transpose(self):
        def s_2(i): return [self.matrix[j][i] for j in range(self.row)]
        return Matrix([s_2(i) for i in range(self.col)])

    def multiply(self):
        pass

def linear(x, bias = [1]):
    x_p = []
    for i, v in enumerate(x):
        x_p.append(sum([1 + v * bias[i] for i in range(len([i]))]))
    return x_p

def cost(hyp, y):
    total = len(hyp)
    return 1/(2*total)*sum([math.pow(hyp[i] - y[i], 2) for i in range(total)])

def cost_der(hyp, y):
    total = len(hyp)
    return 1/total*sum([hyp[i] - y[i] for i in range(total)])

def grad_descent(loss):
    temp0 = loss - const

def normal_eq():
    pass

def main():
    # Matrix 
    # A[0][0]B[0][0] * A[0][1]B[1][0] * A[0][2]B[2][0]
    # A[0][0]B[0][1] * A[0][1]B[1][1] * A[0][2]B[2][1]
    # A[0][0]B[0][2] * A[0][1]B[1][2] * A[0][2]B[2][2]
    # A[0][0]B[0][3] * A[0][1]B[1][3] * A[0][2]B[2][3]
    
    # Linear
    pop = [449.48, 553.57, 696.783, 870.133, 1000.4, 1309.1]
    year = [1960, 1970, 1980, 1990, 2000, 2010]
    A = Matrix([1] * 6, pop)
    # Sini
    lin = linear(pop_p)
    loss = cost(lin, pop)
    print(loss)


if __name__ == '__main__':
    main()
