import os
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

# Modify these paths
ANNOTATIONS_DIR = "C:\\Users\\Nayanaa\\Desktop\\minip\\dataset\\Annotations"
IMAGES_DIR = "C:\\Users\\Nayanaa\\Desktop\\minip\\dataset\\Annotations"
TFRECORD_FILE = "train.record"

def xml_to_tfrecord(annotation_dir, image_dir, tfrecord_file):
    writer = tf.io.TFRecordWriter(tfrecord_file)
    
    # Class name to class ID mapping
    class_name_to_id = {
        'roller': 1,
        'car': 2,
        'phone': 3
    }
    
    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(annotation_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        image_path = os.path.join(image_dir, filename)

        # Load image
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_image = fid.read()

        # Image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Extract bounding boxes and labels
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        classes_text, classes = [], []

        for obj in root.findall('object'):
            name = obj.find('name').text
            classes_text.append(name.encode('utf8'))
            
            # Map class names to their IDs
            class_id = class_name_to_id.get(name, None)
            if class_id is not None:
                classes.append(class_id)
            else:
                print(f"Warning: Class '{name}' not found in the label map!")

            bndbox = obj.find('bndbox')
            xmins.append(float(bndbox.find('xmin').text) / width)
            xmaxs.append(float(bndbox.find('xmax').text) / width)
            ymins.append(float(bndbox.find('ymin').text) / height)
            ymaxs.append(float(bndbox.find('ymax').text) / height)

        # Create TFRecord
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_image),
            'image/format': dataset_util.bytes_feature(b'jpeg'),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        writer.write(tf_example.SerializeToString())
    
    writer.close()

xml_to_tfrecord(ANNOTATIONS_DIR, IMAGES_DIR, TFRECORD_FILE)
print(f"TFRecord saved at {TFRECORD_FILE}")
