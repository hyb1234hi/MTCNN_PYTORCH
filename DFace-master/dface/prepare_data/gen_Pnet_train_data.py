import argparse
import numpy as np
import cv2
import os
import numpy.random as npr
from dface.core.utils import IoU
import dface.config as config

def gen_pnet_data(data_dir,anno_file,prefix):

    neg_save_dir =  os.path.join(data_dir,"12/negative")
    pos_save_dir =  os.path.join(data_dir,"12/positive")
    part_save_dir = os.path.join(data_dir,"12/part")

    for dir_path in [neg_save_dir,pos_save_dir,part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    save_dir = os.path.join(data_dir,"pnet")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    post_save_file = os.path.join(config.ANNO_STORE_DIR,config.PNET_POSTIVE_ANNO_FILENAME)
    neg_save_file = os.path.join(config.ANNO_STORE_DIR,config.PNET_NEGATIVE_ANNO_FILENAME)
    part_save_file = os.path.join(config.ANNO_STORE_DIR,config.PNET_PART_ANNO_FILENAME)
    # 打开保存pos,neg,part文件名、标签的txt文件，这三个是生成文件
    f1 = open(post_save_file, 'w')
    f2 = open(neg_save_file, 'w')
    f3 = open(part_save_file, 'w')
    # 打开原始图片标注txt文件
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    num = len(annotations)
    print("%d pics in total" % num)
    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # part
    idx = 0
    box_idx = 0
    # 原始图片根据标注的bbox，生成negative,posotive,part图片，标注形式也做相应变化
    for annotation in annotations: #逐行读取，按作者的方式，每行为一个原图
        annotation = annotation.strip().split(' ')  #对读取的每一行，按空格进行切片
        im_path = os.path.join(prefix,annotation[0]) # 第1个为图片名
        bbox = list(map(float, annotation[1:])) #第2个到最后一个为bbox
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4) # 对bbox进行reshape，4个一列
        print('prefix',prefix,'annotation[0]',annotation[0])
        img = cv2.imread(im_path)
        idx += 1
        if idx % 100 == 0:
            print(idx, "images done")

        height, width, channel = img.shape

        neg_num = 0
        # 生成nagative，每个原图生成50个negative sample
        while neg_num < 50:
            # size表示neg样本大小，在12和min(width, height)/2之间随机取一个整数
            #***********************************************************
            size = npr.randint(12, min(width, height) / 2)
            # neg的左上角坐标(x1,y1)，在0和(width - size)之间随机取一个整数
            nx = npr.randint(0, width - size)  #########
            ny = npr.randint(0, height - size)
            # 随机生成的bbox位置(x1,y1),(x2,y2)
            crop_box = np.array([nx, ny, nx + size, ny + size])########
            # 计算随机生成的bbox和原图中所有标注bboxs的交并比
            Iou = IoU(crop_box, boxes)
            # 在原图中crop对应的区域图片，作为negative sample
            cropped_im = img[ny : ny + size, nx : nx + size, :]   #？ 这是具体总裁剪那部分区域没有搞懂
            # 对crop的图像进行resize，大小为12*12  这个12就是resize r-net就设置为24 o-net为48
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            # 如果crop_box与所有boxes的Iou都小于0.3，那么认为它是nagative sample
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                # 保存图片的地址和图片名
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write(save_file + ' 0\n')
                #保存负样本图片
                cv2.imwrite(save_file, resized_im)

                n_idx += 1
                neg_num += 1


        for box in boxes:    #逐行读取，每次循环处理一个box
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            # 生成 positive examples and part faces
            # generate negative examples that have overlap with gt
            for i in range(5):
                # size表示随机生成样本的大小，在12和 min(width, height) / 2之间
                size = npr.randint(12,  min(width, height) / 2)  ## 跟上文生成的size 有冗余
                ## 这里提出的改进思想是 能不能把随机生成的size的样本保存在文件中
                # 在这里直接进行读取 防止再次重新生成心得随机样本造成模型速度过慢

                # delta_x and delta_y are offsets of (x1, y1)
                # delta 表示相对于标注box center的偏移量
                delta_x = npr.randint(max(-size, -x1), w)   ##
                delta_y = npr.randint(max(-size, -y1), h)
                # nx,ny表示偏移后的box坐标位置
                nx1 = int(max(0, x1 + delta_x))   ##
                ny1 = int(max(0, y1 + delta_y))


                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny1 : ny1 + size, nx1 : nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                # size表示随机生成样本的大小，在int(min(w, h) * 0.8) 和 np.ceil(1.25 * max(w, h)) 之间
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                # delta 表示相对于标注box center的偏移量
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                # nx,ny表示偏移后的box坐标位置
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))

                nx2 = int(nx1 + size)
                ny2 = int(ny1 + size)

                # 去掉超出原图的box
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                # bbox偏移量的计算，由 x1 = nx1 + float(size)*offset_x1 推导而来，可以参考bounding box regression博客
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                ny1 = int(ny1)
                ny2 = int(ny2)
                nx1 = int(nx1)
                nx2 = int(nx2)
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR) #INTER_LINEAR表示双线性插值

                box_ = box.reshape(1, -1)
                # 0.4<=Iou<0.65的作为part faces
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()



def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dface_traindata_store', dest='traindata_store', help='dface train data temporary folder,include 12,24,48/postive,negative,part,landmark',
                        default='../data/wider/', type=str)
    parser.add_argument('--anno_file', dest='annotation_file', help='wider face original annotation file',
                        default=os.path.join(config.ANNO_STORE_DIR,"wider_origin_anno.txt"), type=str)
    parser.add_argument('--prefix_path', dest='prefix_path', help='annotation file image prefix root path',
                        default='', type=str)




    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gen_pnet_data(args.traindata_store,args.annotation_file,args.prefix_path)
