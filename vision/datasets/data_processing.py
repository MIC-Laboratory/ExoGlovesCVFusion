from os import listdir
# from os.path import isfile, join
import xml.etree.ElementTree as ET

def data_processing(path, extension, name):
    # Get all the imgs in the directory if they end with .jpg and sort them alphabetically
    imgs = sorted([f"{path}/{f}" for f in listdir(path) if isfile(join(path, f)) and f.endswith(extension)])
    imgs_dict = {name: imgs}
    return imgs_dict

def VOC_parser(xml_path):
    # if xml_path == "train_zip/train/JENGA_COURTYARD_S_T_frame_0002_jpg.rf.731c26c1099e953aabb41e5cc70c8eeb.xml":
    #     print(1)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj_name = []
    bndbox_values = []
    objects = root.findall('object')
    if len(objects) != 0:
        for obj in objects:
            name = obj.find('name').text
            obj_name.append(name)
            bndbox = obj.findall('bndbox')
            for box in bndbox:
                xmin = int(box.find('xmin').text)
                ymin = int(box.find('ymin').text)
                xmax = int(box.find('xmax').text)
                ymax = int(box.find('ymax').text)
                bndbox_values.append([xmin, ymin, xmax, ymax])

    return obj_name, bndbox_values

def data_process_voc_parse(path):
    # Get imgs
    files = sorted([f"{path}/{f}" for f in listdir(path)])
    classes = ["apple","banana","orange"]
    # imgs_dict = {name: imgs}
    obj_name = []
    bndbox_values = []
    imgs = []
    for f in files:
        # if f == "train_zip/train/orange_9.jpg":
        #     print(1)
        # if file is xml, check xml for bndbox; if no bndbox, skip
        if '.xml' in f:
            tree = ET.parse(f)
            root = tree.getroot()
            objects = root.findall('object')
            if len(objects) != 0:
                for obj in objects:
                    tmp = []
                    tmp2 = []
                    name = obj.find('name').text
                    if name not in classes:
                        continue
                    tmp.append(name)
                    bndbox = obj.findall('bndbox')
                    for box in bndbox:
                        xmin = int(box.find('xmin').text)
                        ymin = int(box.find('ymin').text)
                        xmax = int(box.find('xmax').text)
                        ymax = int(box.find('ymax').text)
                    tmp2.append([xmin, ymin, xmax, ymax])
                bndbox_values.append(tmp2)
                obj_name.append(tmp)
            if len(objects) == 0:
                imgs.pop(-1)
        if '.jpg' in f:
            imgs.append(f)
    return bndbox_values, obj_name, imgs
   