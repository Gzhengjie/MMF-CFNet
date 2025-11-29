'''
日期: 2021年10月07
根据txt文件构建dataset
'''

import os
import shutil

dataset_path = r"G:\TBD-sent"

path_label = dataset_path + r'\label' # labels位置
path_image = dataset_path + r'\image'   #image位置
# dataset_path = r'F:\dataset\qzgy\LSD\256'  # 需保存的路径
test = True
#

if test != True:
    # 只构建训练集和验证集
    with open(dataset_path + r"\train.txt", 'r') as stream:
        lines = stream.readlines()

        if not os.path.exists(dataset_path + r'\train_image'):  # 判断所在目录下是否有该文件名的文件夹
            os.mkdir(dataset_path + r'\train_image')  # 创建多级目录用mkdirs，单击目录mkdir
        if not os.path.exists(dataset_path + r'\train_label'):  # 判断所在目录下是否有该文件名的文件夹
            os.mkdir(dataset_path + r'\train_label')  # 创建多级目录用mkdirs，单击目录mkdir

        for line in lines:
            line = line.replace("\n", "")
            shutil.copy(os.path.join(path_label, line + '.tif'),
                        os.path.join(dataset_path + '/' + 'train_label', line + '.tif'))

            shutil.copy(os.path.join(path_image, line + '.tif'),
                        os.path.join(dataset_path + '/' + 'train_image', line + '.tif'))

    with open(dataset_path + r"\val.txt", 'r') as stream:
        lines = stream.readlines()

        if not os.path.exists(dataset_path + r'\val_image'): #判断所在目录下是否有该文件名的文件夹
            os.mkdir(dataset_path + r'\val_image') #创建多级目录用mkdirs，单击目录mkdir
        if not os.path.exists(dataset_path + r'\val_label'): #判断所在目录下是否有该文件名的文件夹
            os.mkdir(dataset_path + r'\val_label') #创建多级目录用mkdirs，单击目录mkdir


        for line in lines:
            line = line.replace("\n", "")
            shutil.copy(os.path.join(path_label, line + '.tif'),
                        os.path.join(dataset_path + '/' + 'val_label', line + '.tif'))

            shutil.copy(os.path.join(path_image, line + '.tif'),
                        os.path.join(dataset_path + '/' + 'val_image', line + '.tif'))
else:
    # # 构建训练集、验证集、测试集
    # with open(dataset_path + r"\train.txt", 'r') as stream:
    #     lines = stream.readlines()
    #
    #     if not os.path.exists(dataset_path + r'\train_image'):  # 判断所在目录下是否有该文件名的文件夹
    #         os.mkdir(dataset_path + r'\train_image')  # 创建多级目录用mkdirs，单击目录mkdir
    #     if not os.path.exists(dataset_path + r'\train_label'):  # 判断所在目录下是否有该文件名的文件夹
    #         os.mkdir(dataset_path + r'\train_label')  # 创建多级目录用mkdirs，单击目录mkdir
    #
    #     for line in lines:
    #         line = line.replace("\n", "")
    #         shutil.copy(os.path.join(path_label, line + '.tif'),
    #                     os.path.join(dataset_path + '/' + 'train_label', line + '.tif'))
    #
    #         shutil.copy(os.path.join(path_image, line + '.tif'),
    #                     os.path.join(dataset_path + '/' + 'train_image', line + '.tif'))
    #
    # with open(dataset_path + r"\val.txt", 'r') as stream:
    #     lines = stream.readlines()
    #
    #     if not os.path.exists(dataset_path + r'\val_image'):  # 判断所在目录下是否有该文件名的文件夹
    #         os.mkdir(dataset_path + r'\val_image')  # 创建多级目录用mkdirs，单击目录mkdir
    #     if not os.path.exists(dataset_path + r'\val_label'):  # 判断所在目录下是否有该文件名的文件夹
    #         os.mkdir(dataset_path + r'\val_label')  # 创建多级目录用mkdirs，单击目录mkdir
    #
    #     for line in lines:
    #         line = line.replace("\n", "")
    #         shutil.copy(os.path.join(path_label, line + '.tif'),
    #                     os.path.join(dataset_path + '/' + 'val_label', line + '.tif'))
    #
    #         shutil.copy(os.path.join(path_image, line + '.tif'),
    #                     os.path.join(dataset_path + '/' + 'val_image', line + '.tif'))

    with open(dataset_path + r"\test.txt", 'r') as stream:
        lines = stream.readlines()

        if not os.path.exists(dataset_path + r'\test_image'):  # 判断所在目录下是否有该文件名的文件夹
            os.mkdir(dataset_path + r'\test_image')  # 创建多级目录用mkdirs，单击目录mkdir
        if not os.path.exists(dataset_path + r'\test_label'):  # 判断所在目录下是否有该文件名的文件夹
            os.mkdir(dataset_path + r'\test_label')  # 创建多级目录用mkdirs，单击目录mkdir

        for line in lines:
            line = line.replace("\n", "")
            shutil.copy(os.path.join(path_label, line + '.tif'),
                        os.path.join(dataset_path + '/' + 'test_label', line + '.tif'))

            shutil.copy(os.path.join(path_image, line + '.tif'),
                        os.path.join(dataset_path + '/' + 'test_image', line + '.tif'))
            # os.remove(os.path.join(path_label, line + '.tif'))
            # os.remove(os.path.join(path_image, line + '.tif'))

