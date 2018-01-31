from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None


DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M=2^27 -1



'''
例：
create_image_lists(’./data‘,10,10)
功能：返回 result{saber:{},archer:{},einzbern{}} 字典的key为data目录下的子文件夹名字(小写且不要带特殊符号)
saber archer einzbern 对应的key均为字典
saber的key字典结构为:{'dir': 'saber','training': training_images,'testing': testing_images,'validation': validation_images}
'''
def create_image_lists(image_dir, testing_percentage, validation_percentage):
  result = {}
  if not gfile.Exists(image_dir):
    print("Image directory not found.")
    return None
  #eg:sub_dirs=x[0]=['./data', './data\\Archer', './data\\Einzbern', './data\\Saber']
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  is_root_dir = True
  for sub_dir in sub_dirs:
    #skip './data'
    if is_root_dir:
      is_root_dir = False
      continue
    file_list = []
    #eg:dir_name='Archer'
    dir_name = os.path.basename(sub_dir)
    file_glob = os.path.join(image_dir, dir_name, '*.jpg')
    file_list.extend(gfile.Glob(file_glob))
    #file_list:all images path for one category now
    if not file_list:
      print('No images found in '+dir_name)
      continue
    if len(file_list) < 20:
      print('WARNING: Folder has less than 20 images')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      print('WARNING: Folder {} has more than {} images. Some images will never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    #re.sub：正则表达式 去除分类名称中的特殊符号，且将类别名小写
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      # % 取余 计算后的值随机分布在1~100之间，看做均等概率的话，用validation_percentage完成分割
      percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) * (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name) #0 ~ validation_percentage
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)  #validation_percentage ~ testing_percentage+ validation_percentage
      else:
        training_images.append(base_name)  #testing_percentage+ validation_percentage ~ 100
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result


'''
返回指定图片的全路径
'''
def get_image_path(image_lists, label_name, index, image_dir, category):
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


'''
返回瓶颈文件路径=./bottlenecks/**类别**/图片名+.txt
'''
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,category):
  return get_image_path(image_lists, label_name, index, bottleneck_dir,category) + '.txt'


'''
载入 Inception v3 模型(./inception/classify_image_graph_def.pb)
返回 
    graph：Inception v3 模型图
    bottleneck_tensor：经过前面5个卷积2个最大池化三个Inception模块组最后进行8*8平均池化处理后的维度为 1*1*2048 的 tensor
    jpeg_data_tensor：输入层的 299*299*3 tensor
    resized_input_tensor：

'''
def create_inception_graph():
  with tf.Graph().as_default() as graph:
    model_filename = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              'pool_3/_reshape:0', 'DecodeJpeg/contents:0',
              'ResizeBilinear:0']))
  return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


'''
将指定图片数值输入Inception v3 模型的输入层
得到 1*1*2048 层的输出数据
去掉1*1维度 返回 2048 个数值
'''
def run_bottleneck_on_image(sess, image_data, jpeg_data_tensor,bottleneck_tensor):
  bottleneck_values = sess.run(bottleneck_tensor,{jpeg_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


'''
如果 model_dir 下没有 inception-2015-12-05.tgz 文件则下载
递归解压 model_dir 下的所有压缩包       
'''
def maybe_download_and_extract():
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


'''
确认 dir 存在
不存在就创建一个
'''
def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)



'''
读取指定图片数据给下面的函数传参
调用 run_bottleneck_on_image() 拿到 瓶颈层的输出 2048 个值
往 ./bottlenecks/**分类名**/图片名.txt 中写数据
'''
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,image_dir, category, sess, jpeg_data_tensor,bottleneck_tensor):
  print('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  bottleneck_values = run_bottleneck_on_image(sess, image_data,jpeg_data_tensor,bottleneck_tensor)
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


'''
如果还没创建文件则调用 create_bottleneck_file() 生成瓶颈层输出文件
如果创建了文件则读取 瓶颈层输出文件 拿到瓶颈层输出 2048 个节点的数值放入 bottleneck_values[] 并返回
'''
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,category, bottleneck_dir, jpeg_data_tensor,bottleneck_tensor):
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,bottleneck_dir, category)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,image_dir, category, sess, jpeg_data_tensor,bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,image_dir, category, sess, jpeg_data_tensor,bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


'''
通过循环调用 get_or_create_bottleneck()
遍历各个类别的 训练 测试 验证 集
生成所有图片的 瓶颈层数据
'''
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,jpeg_data_tensor, bottleneck_tensor):
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(sess, image_lists, label_name, index,image_dir, category, bottleneck_dir,jpeg_data_tensor, bottleneck_tensor)
        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          print(str(how_many_bottlenecks) + ' bottleneck files created.')


'''
category 指定返回 training/testing/validation 哪个
如果 FLAGS.train_batch_size 默认的话返回所有指定的图片的瓶颈层输出列表
如果 FLAGS.train_batch_size >=0的话返回瓶颈层输出列表(每个类别张数都等于tbs,随机选择)
返回三个 bottlenecks：瓶颈层输出列表 ground_truths：答案标签列表 filenames：图片全路径列表
'''
def get_random_cached_bottlenecks(sess, image_lists, how_many, category,bottleneck_dir, image_dir, jpeg_data_tensor,bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index,image_dir, category)
      bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,image_index, image_dir, category,bottleneck_dir, jpeg_data_tensor,bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,image_dir, category)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,image_index, image_dir, category,bottleneck_dir, jpeg_data_tensor,bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames


'''
category 为固定值 training 只在训练的时候调用
FLAGS.train_batch_size  不能为默认值，必须大于0
随机取图变形然后拿到他的2048瓶颈层特征，不保存为文件，只放入bottlenecks列表
返回两个 bottlenecks：瓶颈层输出列表 ground_truths：答案标签列表
'''
def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,distorted_image, resized_input_tensor, bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    distorted_image_data = sess.run(distorted_image,{input_jpeg_tensor: jpeg_data})
    bottleneck = run_bottleneck_on_image(sess, distorted_image_data,resized_input_tensor,bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


'''

功能：
     判断是否要做数据增强 4 个有一个不为 0 就返回 True
     flip_left_right | random_crop | random_scale | random_brightness
         左右翻转         随机切割       随机缩放           随机打光

'''
def should_distort_images(flip_left_right, random_crop, random_scale,random_brightness):
  return (flip_left_right or (random_crop!=0) or (random_scale!=0) or (random_brightness!=0))


'''
返回占位符和变形后的数据
'''
def add_input_distortions(flip_left_right, random_crop, random_scale,random_brightness):
  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),minval=1.0,maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
  precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precropped_image_3d,[MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),minval=brightness_min,maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result


'''
方便用 tensorflow 查看训练过程
'''
def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


'''
最后连接一层全连接层 weight：[2048,类别数目]
定义好损失函数 优化方式
返回各项数值
'''
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,[None, class_count],name='GroundTruthInput')

  with tf.name_scope('final_training_ops'):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],
                                          stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


'''
返回 预测正确率 和 预测可能性最大的那个名字
'''
def add_evaluation_step(result_tensor, ground_truth_tensor):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_or_not = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_or_not, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction



'''
删除  summaries_dir  目录的所有文件
调用  maybe_download_and_extract() 下载模型
调用  create_inception_graph() 加载图结构
调用  create_image_lists() 得到result字典
调用  should_distort_images() 判断是否做数据增强 做的话调用  add_input_distortions() 得到变形后的数据
调用  cache_bottlenecks()  将图片转换为瓶颈层处理后的输出 tensor 保存 直接用以训练后面的分类层
调用  add_final_training_ops() 设置最后一层的相关参数
调用  add_evaluation_step() 得到平均误差 最大预测者
将所有的 summary 写入 /tmp/retrain_logs/ 目录
初始化参数
如果不用数据增强的话则调用  get_random_cached_bottlenecks() 拿到瓶颈层输出列表和答案标签列表
如果需要数据增强的话则调用  get_random_distorted_bottlenecks() 拿到瓶颈层输出列表和答案标签列表
将瓶颈层输出列表和答案标签列表分别作为输入数据和答案去训练最后一层 显示各项数据
训练完成用tesing的图片去做最后一次测试，输出正确率
判断  FLAGS.print_misclassified_test_images 决定是否要显示没法分类的图片名字
将训练完成的图写入 ./retrained_graph.pb 
将分类标签写入 ./retrained_labels.txt 
'''
def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  maybe_download_and_extract()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,FLAGS.validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0:
    print('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    print('Only one valid folder of images found at ' + FLAGS.image_dir +
          ' - multiple classes are needed for classification.')
    return -1

  do_distort_images = should_distort_images(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,FLAGS.random_brightness)

  with tf.Session(graph=graph) as sess:

    if do_distort_images:
      (distorted_jpeg_data_tensor,distorted_image_tensor) = add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop,FLAGS.random_scale, FLAGS.random_brightness)
    else:
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,FLAGS.bottleneck_dir, jpeg_data_tensor,bottleneck_tensor)

    (train_step, cross_entropy, bottleneck_input, ground_truth_input,final_tensor) = add_final_training_ops(class_count,FLAGS.final_tensor_name,bottleneck_tensor)

    evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    init = tf.global_variables_initializer()
    sess.run(init)


    for i in range(FLAGS.how_many_training_steps):
      if do_distort_images:
        (train_bottlenecks,train_ground_truth) = get_random_distorted_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training',FLAGS.image_dir, distorted_jpeg_data_tensor,distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        (train_bottlenecks,train_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training',FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,bottleneck_tensor)

      train_summary, _ = sess.run([merged, train_step],feed_dict={bottleneck_input: train_bottlenecks,ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy],feed_dict={bottleneck_input: train_bottlenecks,ground_truth_input: train_ground_truth})
        print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,train_accuracy * 100))
        print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (get_random_cached_bottlenecks(sess, image_lists, FLAGS.validation_batch_size, 'validation',FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,bottleneck_tensor))
        validation_summary, validation_accuracy = sess.run([merged, evaluation_step],feed_dict={bottleneck_input: validation_bottlenecks,ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %(datetime.now(), i, validation_accuracy * 100,len(validation_bottlenecks)))


    test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,'testing', FLAGS.bottleneck_dir,FLAGS.image_dir, jpeg_data_tensor,bottleneck_tensor))
    test_accuracy, predictions = sess.run([evaluation_step, prediction],feed_dict={bottleneck_input: test_bottlenecks,ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
      print('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():print('%70s  %s' % (test_filename,list(image_lists.keys())[predictions[i]]))


    output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')


'''
第一个运行的函数
功能：获取各项参数，调用main函数
'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='./data',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='./retrained_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='./retrained_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='./model',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='./bottlenecks',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=0,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)