import tensorflow as tf
import glob

tfrecord_dir = "./modified_libero_rlds/libero_object_no_noops/1.0.0"
tfrecord_files = sorted(glob.glob(f"{tfrecord_dir}/*.tfrecord-*"))

for tfrecord in tfrecord_files:
    try:
        for _ in tf.data.TFRecordDataset(tfrecord).take(1):
            pass
        print(f"✅ OK: {tfrecord}")
    except Exception as e:
        print(f"❌ ERROR in {tfrecord}: {e}")
