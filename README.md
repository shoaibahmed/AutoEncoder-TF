# AutoEncoder-TF

Implementation of Dense and Convolutional Autoencoder in TensorFlow. 
<br/>Different models can be found in the autoencoder.py file. New models can be added to the same file by following the format defined for other models.
<br/>Use <b>trainer_ae.py</b> for training with data loading mechanism implemented in Numpy and <b>trainer_ae_queues.py</b> for using the TensorFlow Queueing mechanism for input data loading.

<br/>TensorFlow Dataset API has been utilized for data reading in the latest version of the trainer i.e. <b>trainer_ae_latest.py</b>.
<br/>To start training of convolutional auto-encoder from scratch along with tensorboard visualization, use the following command:
'''
python ./trainer\_ae\_latest.py -t -s --tensorboardVisualization --convolutionalAutoencoder --batchSize 50 --logsDir ./logs/ --modelDir ./model/
'''
Similarly, for testing along with computation of the output image files, use the command:
'''
python ./trainer\_ae\_latest.py -c --computeOutputImages --imagesOutputDirectory ./AutoEnc-output/
'''

<br/><br/> Author: <b>Shoaib Ahmed Siddiqui</b>
<br/> Email: <b>12bscsssiddiqui@seecs.edu.pk</b>
