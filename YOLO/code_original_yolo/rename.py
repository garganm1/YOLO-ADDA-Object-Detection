import glob
import os
# tt = glob.glob("./data/train/images/*.jpg")


# save_path = "./data/"

# name_of_file = "train"

# completeName = os.path.join(save_path, name_of_file+".txt")   

# file1 = open(completeName, "w")

# file1.write(tt[0])

# for j in tt[1:]:
#   file1.write("\n"+j)

# file1.close()


def label_converter(path_to_files, label_to_convert_to):

  txt = glob.glob(path_to_files)

  for f in txt:

    with open(f) as file_val:
      attrb = file_val.read().split()
    attrb[0] = label_to_convert_to

    attrb = attrb[0]+" "+attrb[1]+" "+attrb[2]+" "+attrb[3]+" "+attrb[4]

    my_file = open(f, "w")

    my_file.write(attrb)

    my_file.close()

label_converter('./data/validation/labels/shuriken/*.txt', '5')