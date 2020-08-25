import random

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
        return self
    
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

    def inverse(self):
        if self.row == 2 and self.col == 2:
            temp = self.matrix[0][0]
            self.matrix[0][0] = self.matrix[1][1]
            self.matrix[1][1] = temp
            self.matrix[0][1] = -self.matrix[0][1]
            self.matrix[1][0] = -self.matrix[1][0]
            determinant = self.matrix[0][0]*self.matrix[1][1] - self.matrix[0][1]*self.matrix[1][0]
            return self.multiply([1/determinant])

    def multiply(self, matrix):
        for i in range(self.row):
            for j in range(self.col):
                if len(matrix) == 1:
                    self.matrix[i][j] *= matrix[0]
                else:
                    self.matrix[i][j] = matrix[i][j]
        return self

def linear(x, param):
    return [param[i][0] + sum([x[i][j] * param[i][j+1] for j in range(len(x[i]))]) for i in range(len(x))]

def cost(x, y, param):
    n = len(y)
    x = linear(x, param)
    return 1/(2*n)*sum([(x[i] - y[i])**2 for i in range(n)])

def cost_der(x, y, param, active = 0):
    n = len(y)
    x = linear(x, param)
    return 1/n*sum([x[i]*param[i][active] for i in range(n)])

def grad_descent(x, y, param, lrate = 0.1, target = 0.5):
    loss = cost_der(x, y, param, 0)
    der = [loss - (lrate*cost_der(x, y, param, i)) for i in range(len(param[0]))]
    print(der)
    if target <= loss:
        grad_descent(x, y, [der] * len(x), lrate = lrate, target = target)

def normal_eq():
    pass

def main():
    # Matrix 
    # A[0][0]B[0][0] * A[0][1]B[1][0] * A[0][2]B[2][0]
    # A[0][0]B[0][1] * A[0][1]B[1][1] * A[0][2]B[2][1]
    # A[0][0]B[0][2] * A[0][1]B[1][2] * A[0][2]B[2][2]
    # A[0][0]B[0][3] * A[0][1]B[1][3] * A[0][2]B[2][3]
    # Linear
    pop = [1,2,3,4,5,6]
    year = [[1], [2], [3], [4], [5], [6]]
    grad_descent(year, pop, [[1, 1]] * len(year), lrate = 0.1, target = 1)
    # Sini
    


if __name__ == '__main__':
    main()
