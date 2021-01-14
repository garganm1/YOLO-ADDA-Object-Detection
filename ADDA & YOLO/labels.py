import glob


def label_converter(path_to_files, label_to_convert_to):

  txt = glob.glob(path_to_files)
 

  if txt is None:
    print("Empty")

  for f in txt:

    with open(f) as file_val:
      attrb = file_val.read().split()
    attrb[0] = label_to_convert_to

    attrb = attrb[0]+" "+attrb[1]+" "+attrb[2]+" "+attrb[3]+" "+attrb[4]

    my_file = open(f, "w")

    my_file.write(attrb)

    my_file.close()

label_converter('./labels/Phone*.txt', '2')