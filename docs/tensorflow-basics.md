# Tensorflow Basics

## Tensorflow 1.0

The standard worflow for Tensorflow 1.0 is:

- Step 1: Create the tf.Graph object and set it as the default graph for the current scope.
- Step 2: Describe the computation using the Tensorflow API (e.g. y = tf.matmul(a,x) + b).
- Step 3: Think in advance about variable sharing and define the variables’ scope accordingly.
- Step 4: Create and configure the tf.Session.
- Step 5: Build the concrete graph and load it into the tf.Session.
- Step 6: Initialize all the variables.
- Step 7: Use the tf.Session.run method to start the computation. The node execution will trigger a backtracking procedure from the chosen nodes (.run input parameters) to their inputs, in order to resolve the dependencies and compute the result.

Example of Tensorflow 1.0:

```python
g = tf.Graph()
with g.as_default():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(y)
```

## Tensorflow 2.0

But in Tensorflow 2.0, most of the code that uses `Placeholders`, `Sessions`,... are from Tensorflow 1.0 and are removed in Tensorflow 2.0:

- Remove the graph definition.
- Remove the session execution.
- Remove variables initialization.
- Remove the variable sharing via scopes.
- Remove the tf.control_dependencies to execute sequential operation not connected by a dependency relation.

Example of Tensorflow 2.0:

```python
a = tf.constant([[10,10],[11.,1.]])
x = tf.constant([[1.,0.],[0.,1.]])
b = tf.Variable(12.)
y = tf.matmul(a, x) + b
y.numpy()
```

## TF1.0 Variable Scope

Variable sharing is a mechanism in TensorFlow that allows for sharing variables accessed in different parts of the code without passing references to the variable around. `tf.variable_scope()` manages namespaces for names passed to `tf.get_variable()`.

```python
with tf.variable_scope("scope"):
    v1 = tf.get_variable("v1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="v2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # scope/v1:0
print(v2.name)  # scope/v2:0
print(a.name)   # scope/Add:0
```

## TF1.0 Get Variable

It is convenient to use the tf.get_variable() function instead of tf.Variable(). The function tf.get_variable() returns the existing variable with the same name if it exists, and creates the variable with the specified shape and initializer if it does not exist.

## TF1.0 Placeholder

Normally we run like this to start training 

```python
x = tf.constant(2)
y = x * 42
with tf.Session() as sess:
  sess.run(y) // 84
```

But if you use placeholder, we will need to pass the data into `feed_dict` before we can actually start training.

```python
x = tf.placeholder(tf.float32)
y = x * 42
with tf.Session() as sess:
  sess.run(y, feed_dict={x: 2}) // 84
```

## TF1.0 Embedding Lookup

TBD

## TF1.0 Global Variables Initializer

TBD

## TF1.0 Train Saver

TBD

## Reference
- https://stackoverflow.com/questions/50497724/tensorflow-when-should-i-use-or-not-use-feed-dict
- https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
- https://medium.irfandanish.com/learning-tensorflow-2-use-tf-function-and-forget-about-tf-session-a8117158edd9
