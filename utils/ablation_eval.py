from model import *
import argparse

# 预测过程
def ablation_eval(model, val_loader, model_path, only_info):
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    model.eval()
    accuracies = []
    losses = []
    for batch_id, data in enumerate(val_loader):
        image = data[0]
        sent = data[1]
        label = data[2]
        # print("图片")
        # print(image)
        # print("文本")
        # print(sent)
        # print("标签")
        # print(label)
        if only_info == 'image':
            sent = paddle.zeros_like(sent)
            # print("文本修改")
            # print(sent)
        elif only_info == 'text':
            image = paddle.zeros_like(image)
            # print("图片修改")
            # print(image)
        logits = model(image, sent)
        loss = paddle.nn.functional.cross_entropy(logits, label)
        acc = paddle.metric.accuracy(logits, label)
        accuracies.append(float(acc))
        losses.append(float(loss))
    avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
    print("[validation] accuracy: {}, data type: {}".format(avg_acc, only_info))
    # print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))

# 创建一个解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--only_info', type=str, choices=['image', 'text', 'full'], default='full')
parser.add_argument('--res_level', type=str, choices=['18', '34', '50'], default='34')
# 解析参数
args = parser.parse_args()
res_level = args.res_level

if res_level == '18':
    model = Res_CNN([2, 2, 2, 2], res_level)
else:
    model = Res_CNN([3, 4, 6, 3], res_level)

ablation_eval(model, val_loader, "../saved_models/model_final.pdparams", args.only_info)


