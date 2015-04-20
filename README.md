# parallel_leaf v0.0
Ensemble learning on IPython.parallel. The codes are under developing state and constantly updated, please check the update if you get any error :). Please fill free to write the [wiki])(https://github.com/wasit7/parallel_forest/wiki).
## Requirements
You just need [IPython.parallel] (http://ipython.org/ipython-doc/dev/parallel/)
## Setup
1. Creating the dataset
  * By **executing spiral_pickle.py** then the data is recorded into a pickle file. You have to rename the **spiral.pic** to **datasetXX.pic**, where [XX] is index of an engine you have to change the index according to the number of engine you have. For example, a number of engines is 8 the number [XX] are 00, 01, 02, ..., 07.
  * The recorded data may be loaded and verified by **executing spiral_pickle_r.py**
2. Starting ipcluster. Here you can check the instruction in the Ipython website. If you are using TU-Scicloud please check the tutorial [here] (https://www.youtube.com/watch?v=7uZpEoLWhb4)
3. When the data and the cluster are ready, we can start training by **running scmain.py**
