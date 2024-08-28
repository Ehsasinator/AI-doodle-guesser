# AI-doodle-guesser
Here is the AI doodle trainer I coded using kerbs and tensorflow to learn from .npy files to learn doodles and to make a .h5 file that can be used to guess users doodles


To add your .npy files, store the .npy files in the same directory as the python code, and to type the name of each category in a category.txt file

https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=true

.npy files for various categories can be found in the link above ^^


model.add(layers.Dense(18, activation = 'softmax'))  in this line change 18 to how many words you have


doodle_model.save("/Users/ehsanjaveddeveloper/Desktop/doodle_model.h5") change this line at the end for whatever you want to save it as 
     
