import numpy as np

labels=np.array([    ['789.jpg', 1]
                    ,['45.jpg', 2]
                    ,['149.jpg', 2]
                    ,['825.jpg', 3]
                    ,['1330.jpg', 2]
                    ,['439.jpg', 3]])
# print(labels)
# 要将label文件转化为tensorflow能够输入的形式
#       转化前的格式为：                  转化后的格式为：
#     [ [‘0.jpg’   1]                    ‘0.jpg’   [ [1 0 0]
#       ['1.jpg'   2]                    ‘0.jpg’     [0 1 0]
#       ['2.jpg'   3] ]                  ‘0.jpg’     [0 0 1] ]

def convert_types_into_seq(labels):
    labels_numtype = [int(x) for x in labels[:,1]]
    total_types = len(set(labels_numtype))
    total_nums = len(labels_numtype)

    init_array = np.zeros(total_types)
    for i in range(total_nums):
        array = np.zeros(total_types)
        array[labels_numtype[i]-1] = 1
        # print(labels_numtype[i])
        if i == 0:
            init_array = array
        else:
            init_array = np.row_stack((init_array,array))
    # print(init_array)
    labels_out=init_array
    # print(type(labels))
    # print(labels)



    return labels_out


# convert_types_into_seq(labels)