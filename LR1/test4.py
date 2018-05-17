import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import random


# Step 1: read in data from the .xls file
DATA_FILE = "F:\PiyushWS\TF_TEST\LR1\data\data.xlsx"
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1
print(data)

# Step 2: create placeholders for input X and label Y 
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Step 3: create weight and bias, initialized to 0
w1 = tf.Variable(0.0, name="weight1")
w2= tf.Variable(0.0, name="weight2")
b = tf.Variable(0.0, name="bias")

# Step 4: construct model to predict Y  from the X
Y_predicted = X * X * w1 + X * w2 + b
#I am adding a random predicted value
#Y_predicted = X*1.2 # random.randint(1,10)

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name="loss")

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess :
      
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
       
    # Step 8: train the model
    for i in range(100): # run 100 epochs
        for x, y in data:
            # Session runs train_op to minimize loss
            sess.run(optimizer, feed_dict={X: x, Y:y})
    
    # Step 9: output the values of w and b
    w1_value, w2_value, b_value = sess.run([w1, w2,  b])

print("Weight1:",w1_value)
print("Weight2:",w2_value)
print("Bias : ",b_value)