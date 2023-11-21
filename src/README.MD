#

# TF Placeholder

```python
x = tf.constant(2)
y = x * 42
with tf.Session() as sess:
  sess.run(y) // 84
```

But if you use placeholder

```python
x = tf.placeholder(tf.float32)
y = x * 42
with tf.Session() as sess:
  sess.run(y, feed_dict={x: 2}) // 84
```
