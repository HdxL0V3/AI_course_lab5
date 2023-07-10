from model import *
import paddle.optimizer.lr as lr
import matplotlib.pyplot as plt
import argparse

# paddle版本
print(paddle.__version__)
# 数据导入情况
print('\ttrain_dataset snapshot\t')
for image, data, label in train_dataset:
    print(data)
    print(np.array(data).shape)
    print(label)
    break
print('\tval_dataset snapshot\t')
for image, data, label in val_dataset:
    print(data)
    print(np.array(data).shape)
    print(label)
    break
print('\ttest_dataset snapshot\t')
for image, data, label in test_dataset:
    print(data)
    print(np.array(data).shape)
    print(label)
    break

# 打印训练过程
def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()

# 训练过程
def train(model, learning_rate, num_epochs, lr_decay, draw):
    model.train()
    if lr_decay:
        # 创建一个学习率调度器
        steps_per_epoch = len(train_loader)
        lr_scheduler = lr.PolynomialDecay(learning_rate, num_epochs * steps_per_epoch, end_lr=0)
        # 然后将lr_scheduler传递给你的优化器
        opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr_scheduler)
    else:
        opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    steps = 0
    Iters, total_loss, total_acc = [], [], []
    for epoch in range(num_epochs):
        for batch_id, data in enumerate(train_loader):
            steps += 1
            image = data[0]
            sent = data[1]
            label = data[2]
            logits = model(image, sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            if batch_id % 50 == 0:
                Iters.append(steps)
                total_loss.append(float(loss))
                total_acc.append(float(acc))
                if lr_decay:
                    print("epoch: {}, batch_id: {}, loss: {}, lr: {}".format(epoch, batch_id, float(loss), lr_scheduler.get_lr()))
                else:
                    print("epoch: {}, batch_id: {}, loss: {}, lr: {}".format(epoch, batch_id, float(loss), learning_rate))
                # total_loss.append(loss.numpy()[0])
                # total_acc.append(acc.numpy()[0])
                # print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()
        # 验证集评估阶段
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(val_loader):
            image = data[0]
            sent = data[1]
            label = data[2]
            logits = model(image, sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)
            accuracies.append(float(acc))
            losses.append(float(loss))
        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))
        model.train()
    paddle.save(model.state_dict(), "../saved_models/model_final.pdparams")
    if draw:
        draw_process("trainning loss", "red", Iters, total_loss, "trainning loss")
        draw_process("trainning acc", "green", Iters, total_acc, "trainning acc")

# 预测过程
def predict(model, test_loader):
    model.eval()
    results = []
    for batch_id, data in enumerate(test_loader):
        image = data[0]
        sent = data[1]
        logits = model(image, sent)
        pred = paddle.argmax(logits, axis=1).numpy()
        results.extend(pred)
    return results

# 创建一个解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--res_level', type=str, choices=['18', '34', '50'], default='34')
parser.add_argument('--learning_rate', type=float, default=0.00005)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--draw', default=False, action='store_true')
# 解析参数
args = parser.parse_args()

res_level = args.res_level
learning_rate = args.learning_rate
num_epochs = args.epoch
lr_decay = args.lr_decay
draw = args.draw

# resnet 和 一维CNN卷积
if res_level == '18':
    model = Res_CNN([2, 2, 2, 2], res_level)
else:
    model = Res_CNN([3, 4, 6, 3], res_level)

train(model, learning_rate, num_epochs, lr_decay, draw)

results = predict(model, test_loader)
label_map = {0: "negative", 1: "neutral", 2: "positive"}
with open('../raw_data/test_without_label.txt', 'r') as f, open('../predictions/predict.txt', 'w') as f_out:
    lines = f.readlines()
    f_out.write(lines[0])  # Write the header
    for result, line in zip(results, lines[1:]):
        guid, _ = line.strip().split(',')
        f_out.write(f'{guid},{label_map[result]}\n')