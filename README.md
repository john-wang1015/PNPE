# PNPE

In order to reproduce BVCBM example, you need compile C++ code in MATLAB by using following steps:
```
1. type this in the commend line of MATLAB: clibgen.generateLibraryDefinition("TumourModel\Model.cpp")
2. uncomment all function and set <SHAPE> = 1
3. remove .xml file and use: build(defineModel) to build the model}
```

then make sure you instal matlabengine by using: `pip install matlabengine`

## For SVAR model
you can use the dataset we provide to run SNPE and PNPE or choose to comment out the dataset and create your own datasets. The unconditional normalizing flow need to use either z-score or robust statistics to perform the normalization so need to empricially test the performance. Clipping will significantly affect the SNPE for this example with increasing waster of training data. 
