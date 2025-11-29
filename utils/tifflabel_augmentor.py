from osgeo import gdal
import numpy as np
import os


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    # return im_data
    return im_data, im_width, im_height


def writeTiff(im_data, im_width, im_height, im_bands, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr


def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr


def mirror_flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.flipud(new_arr)
    return new_arr


def mirror_flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    new_arr = np.flipud(new_arr)
    return new_arr


def mirror_flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    new_arr = np.flipud(new_arr)
    return new_arr

#
# filepath = r"F:\gzj\data\QZGYD\val_label"
# filename = os.listdir(filepath)
# Img_H = 256
#
# for file in filename:
#     tifname = filepath + "/" + file
#     newtifname = file.split(".")[0]
#     print(newtifname)
#
#     tiff = readTif(tifname)
#
#     # 旋转90度 旋转90度后镜像翻转
#     flip190_left = flip90_left(tiff)
#     newtiffile90 = filepath + "/" + newtifname + "_90.tif"
#     writeTiff(flip190_left, Img_H, Img_H, 1, newtiffile90)
#
#     mirror_flip190_left = mirror_flip90_left(tiff)
#     newtiffile901 = filepath + "/" + newtifname + "_901.tif"
#     writeTiff(mirror_flip190_left, Img_H, Img_H, 1, newtiffile901)
#
#     # 旋转180度 旋转180度后镜像翻转
#     flip1180 = flip180(tiff)
#     newtiffile180 = filepath + "/" + newtifname + "_180.tif"
#     writeTiff(flip1180, Img_H, Img_H, 1, newtiffile180)
#
#     mirror_flip1180 = mirror_flip180(tiff)
#     newtiffile1801 = filepath + "/" + newtifname + "_1801.tif"
#     writeTiff(mirror_flip1180, Img_H, Img_H, 1, newtiffile1801)
#
#     # 旋转270度 旋转270度后镜像翻转
#     flip190_right = flip90_right(tiff)
#     newtiffile270 = filepath + "/" + newtifname + "_270.tif"
#     writeTiff(flip190_right, Img_H, Img_H, 1, newtiffile270)
#
#     mirror_flip190_right = mirror_flip90_right(tiff)
#     newtiffile2701 = filepath + "/" + newtifname + "_2701.tif"
#     writeTiff(mirror_flip190_right, Img_H, Img_H, 1, newtiffile2701)
#
#     tiff0 = np.flipud(tiff)
#     newtiffile001 = filepath + "/" + newtifname + "_1.tif"
#     writeTiff(tiff0, Img_H, Img_H, 1, newtiffile001)
