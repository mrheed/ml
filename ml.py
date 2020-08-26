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
                self.matrix[i][j] *= matrix[0] if len(matrix) == 1 else matrix[i][j]
        return self

def linear(x, param):
    # Return fungsi hipotesis
    return [param[0] + sum([x[i][j] * param[j+1] for j in range(len(x[i]))]) for i in range(len(x))]

def cost(x, y, param):
    n = len(y)
    # Hitung prediksi titik y
    x = linear(x, param)
    # Hitung jumlah loss menggunakan MSE (Mean Squared Error)
    return 1/(2*n)*sum([(x[i] - y[i])**2 for i in range(n)])

def cost_der(x, y, param, active = 0):
    n = len(y)
    x = linear(x, param)
    #print([(x[i]-y[i])*(param[i][active] if active != 0 else 1) for i in range(n)])
    # Turunan parsial dari fungsi cost
    return (1/n)*sum([(x[i]-y[i])*(param[active] if active != 0 else 1) for i in range(n)])

# Gradient descent, mencari titik minimum dari fungsi linear hingga gradient dari titik tersebut menjadi 0
# Bukannya semakin turun malah naik (aneh betul)
def grad_descent(x, y, param, lrate = 0.1, epoch = 100, loss = 0):
    der = [param[i] - (lrate*cost_der(x, y, param, i)) for i in range(len(param))]
    print("[Loss] {}, [Param 0] {} [Param 1] {}".format(cost(x,y,param), der[0], der[1]))
    if epoch != 0:
        grad_descent(x, y, der, lrate = lrate, epoch = epoch-1)

# Normal equation
def normal_eq():
    pass

def main():
    # Matrix 
    # A[0][0]B[0][0] * A[0][1]B[1][0] * A[0][2]B[2][0]
    # A[0][0]B[0][1] * A[0][1]B[1][1] * A[0][2]B[2][1]
    # A[0][0]B[0][2] * A[0][1]B[1][2] * A[0][2]B[2][2]
    # A[0][0]B[0][3] * A[0][1]B[1][3] * A[0][2]B[2][3]
    # Linear
    pop = [11,22,33,44,55,66]
    year = [[1], [2], [3], [4], [5], [6]]
    loss = [cost_der(year, pop, [1, 1], i) for i in range(2)]
    grad_descent(year, pop, [0, 2], lrate = 0.01, epoch = 500)
    # Sini
    


if __name__ == '__main__':
    main()
