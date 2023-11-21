# Prerequisite Knowledge

## tf.train.Saver

TBD

## TF Parts

TBD

## TF Variable Scope

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

## TF Embedding Lookup

TBD

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

## Reference
- https://stackoverflow.com/questions/50497724/tensorflow-when-should-i-use-or-not-use-feed-dict
- https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow