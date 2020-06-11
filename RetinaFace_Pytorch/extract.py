import shutil
import zipfile

from utils import ensure_folder


def extract(filename, tgt_folder='data'):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(tgt_folder)
    zip_ref.close()


if __name__ == "__main__":
    tgt_folder = 'data/widerface'
    ensure_folder(tgt_folder)
    extract('data/retinaface_gt_v1.1.zip', tgt_folder)
    extract('data/WIDER_train.zip')
    extract('data/WIDER_val.zip')
    extract('data/WIDER_test.zip')
    shutil.move('data/WIDER_train/images', 'data/widerface/train/')
    shutil.move('data/WIDER_val/images', 'data/widerface/val/')
    shutil.move('data/WIDER_test/images', 'data/widerface/test/')
    shutil.rmtree('data/WIDER_train/')
    shutil.rmtree('data/WIDER_val/')
    shutil.rmtree('data/WIDER_test/')
