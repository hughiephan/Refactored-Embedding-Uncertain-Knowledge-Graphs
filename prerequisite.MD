# Prerequisite Knowledge

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
