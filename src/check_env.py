import sys, platform
import tensorflow as tf, numpy, pandas, PIL, sklearn, matplotlib
print("Python      :", sys.version.split()[0])
print("Platform    :", platform.platform())
print("TensorFlow  :", tf.__version__)
print("NumPy       :", numpy.__version__)
print("Pandas      :", pandas.__version__)
print("Pillow      :", PIL.__version__)
print("scikit-learn:", sklearn.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("GPUs        :", tf.config.list_physical_devices("GPU"))