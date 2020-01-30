import csv
import numpy as np

def reader(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        in_xi = []
        in_y = []
        in_whole = []
        for row in csv_reader:
            if row:
                in_xi.append([float(row[0]), float(row[1])])
                in_y.append(int(float(row[2])))
                in_whole.append([float(row[0]), float(row[1]), float(row[2])])
        print(len(in_xi))
        return np.asarray(in_xi), np.asarray(in_y), np.asarray(in_whole)






        # def reader(file):
        #     with open(file) as csv_file:
        #         csv_reader = csv.reader(csv_file, delimiter=',')
        #         # line_count = 0
        #         in_xi = []
        #         in_y = []
        #         for row in csv_reader:
        #             if row:
        #                 in_xi.append([float(row[0]), float(row[1])])
        #                 in_y.append(int(float(row[2])))
        #                 # line_count += 1
        #         # print(f'Processed {line_count} lines.')
        #         print(f'xi: {len(in_xi)}|{type(in_xi[0][0])}, y: {len(in_y)}|{type(in_y[0])}')
        #
        #         # for i in in_xi:
        #         #     print(f'{i[0]} | {i[1]}')
        #         #
        #         # for i in in_y:
        #         #     print(f'{i}')