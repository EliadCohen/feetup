#training and testing images can be found here: https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip
#tutorial from: https://towardsdatascience.com/train-image-recognition-ai-with-5-lines-of-code-8ed0bdd8d9ba
from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("idenprof")
model_trainer.trainModel(num_objects=10, num_experiments=200, enhance_data=True, batch_size=32, show_network_summary=True)