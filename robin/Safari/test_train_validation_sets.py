#!/opt/anaconda3/bin/python

import sys, getopt
import os, shutil

def main(argv):
   base_dir = '/Users/robinvanderveken/Projects/Safari/Input'
   output_dir = '/Users/robinvanderveken/Projects/Safari/Subsets'
   folders = ['zebra', 'buffalo', 'rhino', 'elephant']
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('test_train_validation_sets.py -d <imagedir> -o <outputdir> -f <folderlist>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test_train_validation_sets.py -d <imagedir> -o <outputdir> -f <folderlist>')
         sys.exit()
      elif opt in ("-d", "--dir"):
         base_dir = arg
      elif opt in ("-o", "--odir"):
         output_dir = arg
      elif opt in ("-f", "--folders"):
         folders = arg
      else:
         print('Using default settings')

   split_files(base_dir,output_dir,folders)

def split_files(base_dir,output_dir,folders):
   animalfigs = {}
   dirs = ['test', 'train', 'validation']

   for animal in folders:
      figlist = [name for name in os.listdir(os.path.join(base_dir, animal)) if name.split('.')[1] == 'jpg']
      animalfigs.update({animal: figlist})

   for dir in dirs:
      os.mkdir(os.path.join(output_dir, dir))
      for animal in folders:
         os.mkdir(os.path.join(output_dir, dir, animal))
         no_figs = len(animalfigs[animal])
         if dir == 'train':
            figlist = animalfigs[animal][:2 * int(no_figs / 4)]
         elif dir == 'test':
            figlist = animalfigs[animal][2 * int(no_figs / 4):3 * int(no_figs / 4)]
         elif dir == 'validation':
            figlist = animalfigs[animal][3 * int(no_figs / 4):]
         for fig in figlist:
            src = os.path.join(base_dir, animal, fig)
            dst = os.path.join(output_dir, dir, animal, fig)
            shutil.copyfile(src, dst)

if __name__ == "__main__":
   main(sys.argv[1:])