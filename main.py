import MNISTCNN
import argparse
from PIL import Image
import pathlib
import os
import glob

path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Prepossessing script')
parser.add_argument('-p', action="store",               # путь до картинки, которую нужно предсказать
                    dest='sample_path',
                    default=glob.glob(path + r'/*.jpg')
                    )
parser.add_argument('-n',                               # использовать ли обученную модель
                    action="store",
                    dest="mode", default='cold')
parser.add_argument('-d',                               # использовать ли видеокарту
                    action="store",
                    dest="device",
                    default='cuda')
parser.add_argument('-w',                               # путь до папки с весами
                    action="store",
                    dest="weights_path",
                    default=path)
parser.add_argument('-sw', action="store",              # сохранять ли веса
                    dest="save_weights",
                    default=False)
parser.add_argument('-s', action="store",               # путь до папки, куда нужно сохранить веса
                    dest="save_weights_path",
                    default=path)
args = parser.parse_args()


model = MNISTCNN.CNN(args.mode, args.device, args.weights_path if args.mode == 'pretrained' else None)
if args.mode == 'cold':
    model.train()
    f1 = model.eval_f1()
    print(f'Model is ready. Current validation f1 score is {f1}.')

if args.save_weights:
    model.set_weights('save', args.save_weights_path)

img = Image.open(*args.sample_path)
label = model.predict(img)

print(f'This number is {label}.')
