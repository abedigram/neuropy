def read_csv(address):

    f = open(address+'.csv', 'r')
    x_y_label = []

    line = ''
    arr = []
    while True:
        line = f.readline()
        arr = []
        if line:
            # [print(i) for i in line[:len(line)-1].split(',')]
            [arr.append(i) for i in line[:len(line) - 1].split(',')]
            x_y_label.append([float(arr[0]), float(arr[1]), int(arr[2])])
        else:
            break

    return x_y_label