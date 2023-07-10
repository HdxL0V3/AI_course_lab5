from model import *
import argparse
# 预测过程
def predict(model, test_loader, model_path):
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
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
# 解析参数
args = parser.parse_args()
res_level = args.res_level

if res_level == '18':
    model = Res_CNN([2, 2, 2, 2], res_level)
else:
    model = Res_CNN([3, 4, 6, 3], res_level)
results = predict(model, test_loader, "../saved_models/model_final.pdparams")
print(results)
label_map = {0: "negative", 1: "neutral", 2: "positive"}
with open('../raw_data/test_without_label.txt', 'r') as f, open('../predictions/predict.txt', 'w') as f_out:
    lines = f.readlines()
    f_out.write(lines[0])  # Write the header
    for result, line in zip(results, lines[1:]):
        guid, _ = line.strip().split(',')
        f_out.write(f'{guid},{label_map[result]}\n')
