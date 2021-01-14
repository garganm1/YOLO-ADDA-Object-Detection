import glob
import os
import shutil

dir_name = 'data/train.txt'

with open(dir_name) as f:
    all_image_dir = f.readlines()

# print (all_image_dir)
usb = []
shuriken = []
hard_disk = []
hard_disk_2 = []
handgun = []
phone = []
battery = []
battery_2 = []
kitchen_knife = []
kitchen_knife_2 = []
cutter_knife = []

for l in all_image_dir:
    if str.lower(l[14:-5]).startswith('kitchen_knife_2_'):
        kitchen_knife_2.append(l.strip())
    elif str.lower(l[14:-5]).startswith('kitchen_knife'):
        kitchen_knife.append(l.strip())
    elif str.lower(l[14:-5]).startswith('battery_2_'):
        battery_2.append(l.strip())    
    elif str.lower(l[14:-5]).startswith('battery'):
        battery.append(l.strip())
    elif str.lower(l[14:-5]).startswith('phone'):
        phone.append(l.strip())
    elif str.lower(l[14:-5]).startswith('usb'):
        usb.append(l.strip())
    elif str.lower(l[14:-5]).startswith('shuriken'):
        shuriken.append(l.strip())    
    elif str.lower(l[14:-5]).startswith('hard_disk_2_'):
        hard_disk_2.append(l.strip()) 
    elif str.lower(l[14:-5]).startswith('hard_disk'):
        hard_disk.append(l.strip())       
    elif str.lower(l[14:-5]).startswith('handgun'):
        handgun.append(l.strip())
    elif str.lower(l[14:-5]).startswith('cutter_knife'):
        cutter_knife.append(l.strip())

    else:
        handgun.append(l.strip())

# print(len(phone)+len(shuriken)+len(usb)+len(hard_disk)+len(hard_disk_2)+len(handgun)+len(cutter_knife)+len(battery)
#         +len(battery_2)+len(kitchen_knife)+len(kitchen_knife_2))

dest = "./data/training/images/"

if not os.path.exists(dest):
    os.makedirs(dest)
if not os.path.exists(dest+'phone'):
    os.makedirs(dest+'phone')
if not os.path.exists(dest+'usb'):
    os.makedirs(dest+'usb')
if not os.path.exists(dest+'shuriken'):
    os.makedirs(dest+'shuriken')
if not os.path.exists(dest+'battery'):
    os.makedirs(dest+'battery')
if not os.path.exists(dest+'battery_2'):
    os.makedirs(dest+'battery_2')
if not os.path.exists(dest+'handgun'):
    os.makedirs(dest+'handgun')
if not os.path.exists(dest+'cutter_knife'):
    os.makedirs(dest+'cutter_knife')
if not os.path.exists(dest+'hard_disk'):
    os.makedirs(dest+'hard_disk')
if not os.path.exists(dest+'hard_disk_2'):
    os.makedirs(dest+'hard_disk_2')
if not os.path.exists(dest+'kitchen_knife'):
    os.makedirs(dest+'kitchen_knife')
if not os.path.exists(dest+'kitchen_knife_2'):
    os.makedirs(dest+'kitchen_knife_2')


dest = "./data/training/labels/"

if not os.path.exists(dest):
    os.makedirs(dest)
if not os.path.exists(dest+'phone'):
    os.makedirs(dest+'phone')
if not os.path.exists(dest+'usb'):
    os.makedirs(dest+'usb')
if not os.path.exists(dest+'shuriken'):
    os.makedirs(dest+'shuriken')
if not os.path.exists(dest+'battery'):
    os.makedirs(dest+'battery')
if not os.path.exists(dest+'battery_2'):
    os.makedirs(dest+'battery_2')
if not os.path.exists(dest+'handgun'):
    os.makedirs(dest+'handgun')
if not os.path.exists(dest+'cutter_knife'):
    os.makedirs(dest+'cutter_knife')
if not os.path.exists(dest+'hard_disk'):
    os.makedirs(dest+'hard_disk')
if not os.path.exists(dest+'hard_disk_2'):
    os.makedirs(dest+'hard_disk_2')
if not os.path.exists(dest+'kitchen_knife'):
    os.makedirs(dest+'kitchen_knife')
if not os.path.exists(dest+'kitchen_knife_2'):
    os.makedirs(dest+'kitchen_knife_2')

def splitter(list_of_files, dest_i, dest_l):

    for idx, f in enumerate(list_of_files):
        label = f[:7]+'labels'+f[13:-3]+'txt'
        # print(f)
        # print(label)
        # break
        shutil.copy(f, dest_i)
        shutil.copy(label, dest_l)

splitter(cutter_knife, './data/training/images/cutter_knife/', './data/training/labels/cutter_knife/')

splitter(usb, './data/training/images/usb/', './data/training/labels/usb/')

splitter(shuriken, './data/training/images/shuriken/', './data/training/labels/shuriken/')

splitter(kitchen_knife, './data/training/images/kitchen_knife/', './data/training/labels/kitchen_knife/')

splitter(kitchen_knife_2, './data/training/images/kitchen_knife_2/', './data/training/labels/kitchen_knife_2/')

splitter(battery, './data/training/images/battery/', './data/training/labels/battery/')

splitter(battery_2, './data/training/images/battery_2/', './data/training/labels/battery_2/')

splitter(handgun, './data/training/images/handgun/', './data/training/labels/handgun/')

splitter(hard_disk, './data/training/images/hard_disk/', './data/training/labels/hard_disk/')

splitter(hard_disk_2, './data/training/images/hard_disk_2/', './data/training/labels/hard_disk_2/')

splitter(phone, './data/training/images/phone/', './data/training/labels/phone/')

