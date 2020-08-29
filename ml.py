import sys, csv, json, argparse
from pdb import set_trace
from multiprocessing import Pool, Process, Queue, set_start_method, get_context

class Matrix:
    def __init__(self, data):
        self.matrix = data
        self.row = len(data)
        self.col = len(data[0])

    def add(self, matrix):
        for i in range(self.row):
            for j in range(self.col):
                self.matrix[i][j] += matrix[0] if len(matrix) == 1 else matrix[i][j]
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
    return 1/(2*n)*sum([(float(x[i]) - float(y[i]))**2 for i in range(n)])

def cost_der(x, y, param, active = 0):
    n = len(y)
    x = linear(x, param)
    # Turunan parsial dari fungsi cost
    return (1/n)*sum([(float(x[i]) - float(y[i]))*(param[active] if active != 0 else 1) for i in range(n)])

# Gradient descent, mencari titik minimum dari fungsi hingga gradient dari titik tersebut mendekati/menjadi 0
def grad_descent(x, y, param, lrate):
    return [param[i] - (lrate*cost_der(x, y, param, i)) for i in range(len(param))]

def inner_train(q, cost, cur_epoch, x, y, p, lrate, epoch):
    sys.stdout.write('\r')
    sys.stdout.write("[{}%] Loss -> {} | Param -> {} ".format(int(cur_epoch/epoch*100),cost(x,y,p), p))
    sys.stdout.flush()
    q.put({'ce': cur_epoch+1, 'gd': grad_descent(x, y, p, lrate)})

def train(x, y, param, lrate = 0.1, epoch = 100):
    global cur_epoch
    cur_epoch = 0
    p = param
    ctx = get_context('spawn')
    q = ctx.Queue()
    for i in range(epoch):
        ps = ctx.Process(target=inner_train, args=(q, cost, cur_epoch, x, y, p, lrate, epoch,)).start()
        pv = q.get()
        p = pv['gd']
        cur_epoch = pv['ce']
    with open('train.data', 'w') as f:
        f.write(','.join(str(v) for v in p))
        f.close()
    return p

def predict(x, param, expected):
    prediction = linear(x, param)
    delta = 0
    for i, v in enumerate(prediction):
        i = i+1
        d = float(v)-float(expected[i-1])
        d = d if d > 0 else -d
        print("="*50)
        print("{}. Prediction   -> {}".format(i, v))
        print("{}. Expected     -> {}".format(i, expected[i-1]))
        print("{}. Delta        -> {}".format(i, d))
        print("="*50)
        delta += d
    print("Delta Mean       -> {}".format(delta/len(prediction)))


# Normal equation
def normal_eq():
    pass

def read_csv(target, normalize = False, skip = [], y_key = '', limit = -1):
    col = []
    x = []
    y = []
    with open(target) as f:
        data = csv.reader(f, delimiter=',')
        for i, v in enumerate(data):
            if i == 0:
                col = v
            else:
                d = {}
                a = []
                for i2, k in enumerate(v):
                    if col[i2] in skip: continue
                    if col[i2] == y_key:
                        y.append(k)
                        continue
                    d[col[i2]] = k
                if normalize:
                    d = normalization(d)
                    a = [d[k] for k in d]
                x.append(a)
            if i > limit and limit != -1:
                break
        f.close()
    return x, y, len(x[0])

def scale(data, scale):
    return float(data)*scale

def normalization(data):
    for k in data:
        if data[k] == '':
            data[k] = 0
        if k == 'Date':
            n_date = sum([int(j) for j in data[k].split('-')])
            data[k] = scale(n_date, 0.0005)
        elif k in ['MAX', 'MIN', 'MEA', 'YR', 'MO', 'DA', 'Precip']:
            data[k] = scale(data[k], 0.02)
        elif k in ['PRCP']:
            data[k] = scale(data[k], 10)
        elif k in ['MaxTemp', 'MinTemp', 'MeanTemp']:
            data[k] = scale(data[k], 0.05)
        else:
            data[k] = scale(data[k], 0.0001)
    return data

def read_n_train():
    sys.setrecursionlimit(20000)
    skip = ['Precip', 
            'WindGustSpd', 
            'MaxTemp',
            'MinTemp',
            'Snowfall', 
            'PoorWeather', 
            'DR', 
            'SPD', 
            'SNF', 
            'SND', 
            'FT', 
            'FB', 
            'FTI', 
            'ITH', 
            'PGT', 
            'TSHDSBRSGF', 
            'SD3', 
            'RHX', 
            'RHN', 
            'RVG', 
            'WTE', 
            'Date',
            'PRCP']
    n_x, n_y, l_p = read_csv('datasets/Summary of Weather.csv', normalize=True, skip=skip, y_key = 'MeanTemp', limit = -1)
    param = [0] + [(i+1)**2 for i in range(l_p)]
    new_param = train(n_x, n_y, param, 
            lrate = 0.001, 
            epoch = 1000)
    prediction = [i for i in n_x]
    expected = [i for i in n_y]
    predict(prediction, new_param, expected)

def only_predict(path):
    param = []
    x = []
    with open(path, 'r') as f:
        x = [normalization(v) for v in json.loads(f.read())]
        f.close()
    with open('train.data', 'r') as f:
        param = [float(v) for v in f.read().split(',')]
        f.close()
    for i, v in enumerate(linear(x, param)):
        print("="*50)
        print("{}. x -> {}".format(i+1, x[i]))
        print("{}. y -> {}".format(i+1, v))
        print("="*50)

def main():
    # Features
    # 'STA': '10001' 
    # 'YR': '42' 
    # 'MO': '7' 
    # 'DA': '1' 
    # 'MAX': '78' 
    # 'MIN': '72' 
    # 'MEA': '75'
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-p', '--predict', default='')
    args = parser.parse_args()
    set_start_method('spawn')
    if args.train:
        read_n_train()
    if args.predict != '':
        only_predict(args.predict)
    


if __name__ == '__main__':
    main()
