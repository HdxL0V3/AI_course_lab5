import os
path2label = {}
with open('../raw_data/train.txt', 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        # print(line.strip())
        path, label = line.split(',')
        if label == "negative\n":
            label = 0
        elif label == "neutral\n":
            label = 1
        else:
            label = 2
        path2label[path] = label

path= "../raw_data/data"
txt_files = [f+".txt" for f in path2label.keys()]
img_files=[f+".jpg" for f in path2label.keys()]
print(txt_files)
print(img_files)
contents = []  # 存储对应的文本内容
images = []  # 存储相应顺序的图片的位置
labels = []
# with open("./data/1.txt", 'r') as f:
#     content = f.read()
#     print(content)
for i in txt_files:
    with open(path + "/" + i, 'r', errors='ignore') as f:
        content = f.read()
        contents.append(content)
        images.append(path + "/" + i[:-4] + ".jpg")
        labels.append(path2label[i[:-4]])
        # print(content)

all_data_path = "./tmp_data/all_data.txt"
with open(all_data_path, 'w') as f:
    for data in contents:
        f.write(data)

# 生成数据字典
def create_dict(data_path, dict_path):
    with open(dict_path, 'w') as f:
        f.seek(0)
        f.truncate()

    dict_set = set()
    # 读取全部数据
    with open(data_path, 'r', errors='ignore') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for s in content:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    end_dict = {"<pad>": i + 1}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))

    print("数据字典生成完成！")

all_data_path = "./tmp_data/all_data.txt"
dict_path = "./tmp_data/dict.txt"
create_dict(all_data_path, dict_path)

# 创建序列化表示的数据
def load_vocab(file_path):
    fr = open(file_path, 'r', encoding='utf8')
    vocab = eval(fr.read())  # 读取的str转换为字典
    fr.close()
    return vocab


def f_write_txt(words, dict_txt, label):
    labs = ""

    for s in words:
        # lab = str(dict_txt[s])
        lab = str(dict_txt.get(s, dict_txt["<unk>"]))
        labs = labs + lab + ','
    labs = labs[:-1]
    labs = labs + '\t\t\t\t\t' + label + '\n'
    return labs


def create_data_list(data_path, train_path, val_path, dict_path):
    dict_txt = load_vocab(dict_path)
    # print(dict_txt)
    with open(data_path, 'r', errors='ignore') as f_data:
        lines = f_data.readlines()
        print(len(lines))
        # print(lines)

    i = 0
    maxlen = 0
    # with open(train_path, 'w', encoding='utf-8') as f_train:
    with open(train_path, 'w', encoding='utf-8') as f_train, open(val_path, 'w', encoding='utf-8') as f_val:
        for line in contents:
            words = line.split('\t')[-1].replace('\n', '')
            # print(words)
            maxlen = max(maxlen, len(words))
            # label = line.split('\t')[0]
            label = str(labels[i])
            labs = f_write_txt(words, dict_txt, label)
            # 每八个数据抽取一个数据用于验证
            if i % 8 == 0:
                f_val.write(labs)
            else:
                f_train.write(labs)
            i += 1
    print("数据列表生成完成！")
    print(maxlen)

train_path = "./tmp_data/train_data.txt"
val_path = "./tmp_data/val_data.txt"
create_data_list(all_data_path, train_path, val_path, dict_path)

# 测试集处理
path2label_t = {}
with open('../raw_data/test_without_label.txt', 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        # print(line.strip())
        path, label = line.split(',')
        if label == "null\n":
             label = -1
        # elif label == "neutral\n":
        #     label = 1
        # else:
        #     label = 2
        path2label_t[path] = label

path= "../raw_data/data"
txt_files_t = [f+".txt" for f in path2label_t.keys()]
img_files_t = [f+".jpg" for f in path2label_t.keys()]
print(txt_files_t)
print(img_files_t)
contents_t = []  # 存储对应的文本内容
images_t = []  # 存储相应顺序的图片的位置
labels_t = []
# with open("./data/1.txt", 'r') as f:
#     content = f.read()
#     print(content)
for i in txt_files_t:
    with open(path + "/" + i, 'r', errors='ignore') as f:
        content_t = f.read()
        contents_t.append(content_t)
        images_t.append(path + "/" + i[:-4] + ".jpg")
        labels_t.append(path2label_t[i[:-4]])
        # print(content_t)

all_test_data_path = "./tmp_data/all_test_data.txt"
with open(all_test_data_path, 'w') as f:
    for data in contents_t:
        f.write(data)

def create_test_list(data_path, test_path, dict_path):
    dict_txt = load_vocab(dict_path)
    # print(dict_txt)
    with open(data_path, 'r', errors='ignore') as f_data_t:
        lines = f_data_t.readlines()
        print(len(lines))
        # print(lines)
    i = 0
    maxlen = 0
    with open(test_path, 'w', encoding='utf-8') as f_test:
        for line in contents_t:
            words = line.split('\t')[-1].replace('\n', '')
            # print(words)
            maxlen = max(maxlen, len(words))
            # label = line.split('\t')[0]
            label = str(labels_t[i])
            labs = f_write_txt(words, dict_txt, label)
            f_test.write(labs)
            i += 1
    print("测试列表生成完成！")
    print(maxlen)

test_path = "./tmp_data/test_data.txt"
create_test_list(all_test_data_path, test_path, dict_path)

