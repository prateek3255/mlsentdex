import tensorflow as tf

# a=tf.constant(1,tf.int64)
# b=tf.constant(3,tf.int64)
# d=tf.placeholder(tf.int64)
# c=a*b+d
#
# sess=tf.Session()
#
# print(sess.run(c,{d:64}))

#reduce y=wx+b

w=tf.Variable([0.3],tf.float32)
b=tf.Variable([0.3],tf.float32)

x=tf.placeholder(tf.float32)
linear_model=w*x+b;
y=tf.placeholder(tf.float32)

squared_delta=tf.square(linear_model-y)
loss=tf.reduce_sum(squared_delta)


optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

    print(sess.run([w,b]))