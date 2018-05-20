# This Code provides the basic implementation of DNN

# Std TF Calls fo printing and logging
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import Data_loader_HDCO

# This code defines some of the hyper parameters which can be passed as ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Load data from Data Loader class and pass the label column name; 
    # This return tf.Datasets
    (train_x,train_y), (test_x, test_y) = Data_loader_HDCO.load_data('Y')

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
    # defined above as input.
    model = tf.estimator.DNNRegressor(
      hidden_units=[42], feature_columns=my_feature_columns, model_dir="/tmp/Reg_Test2_model")

    # Train the model.
    # By default, the Estimators log output every 100 steps.
    model.train(input_fn=lambda:Data_loader_HDCO.train_input_fn(train_x, train_y,
                                                 args.batch_size),
                                                 steps=args.train_steps)

    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=lambda:Data_loader_HDCO.eval_input_fn(test_x, test_y,
                                                 args.batch_size),
                                                 steps=args.train_steps)

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\nRMS error for the test set:{:.0f}"
            .format(average_loss))


    #Prediction of  local data here
    """   
    predict_x = {
        'x': [4.738013833 , 4.08882312, 2.113558065],
        'y': [1.618906654, 0.132222159, 1.00509475]
        }

    predictions = model.predict(
        input_fn=lambda:Data_loader.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    print(predictions)
    """

    
if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)





