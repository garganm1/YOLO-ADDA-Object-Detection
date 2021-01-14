import os
import glob
import shutil
from sklearn.model_selection import train_test_split

def splitter(path_to_files, ratio = 0.125, rand_stat=42):

  b = glob.glob(path_to_files)

  train_x, test_x = train_test_split(b, test_size=ratio, random_state=rand_stat)
  
  # print(path_to_files[24:-5])
  add_folder = path_to_files[29:-5]
#   print(add_folder)?
#   input()
  dest_train_images = "./train/images/"+add_folder
  dest_train_labels = "./train/labels/"+add_folder
  if not os.path.exists(dest_train_images):
      os.makedirs(dest_train_images)
  if not os.path.exists(dest_train_labels):
    os.makedirs(dest_train_labels)
  for f in enumerate(train_x, 1):
    c = f[1][:-3]+'txt'
    shutil.copy(f[1], dest_train_images)
    shutil.copy(c, dest_train_labels)

  dest_test_images = "./test/images/"+add_folder
  dest_test_labels = "./test/labels/"+add_folder
  
  if not os.path.exists(dest_test_images):
      os.makedirs(dest_test_images)
  if not os.path.exists(dest_test_labels):
    os.makedirs(dest_test_labels)
  for f in enumerate(test_x, 1):
    c = f[1][:-3]+'txt'
    shutil.copy(f[1], dest_test_images)
    shutil.copy(c, dest_test_labels)

splitter('./Orig_n_Aug_Comb/Orig_n_Aug/Hard_disk/*.jpg')