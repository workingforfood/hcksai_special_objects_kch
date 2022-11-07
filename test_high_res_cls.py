import cv2
import numpy as np
from imgaug import augmenters as iaa

from models_base import efficientnet_classifierB2


if __name__=="__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_path = "/var/data2/datasets/hacksai/test"
    test_list_filename = "/var/data2/datasets/hacksai/test.csv"
    model_path = "chkpts_efB2_balanced_ce/base_efB2_balanced_ce_30_0.9992704280155642.h5"
    save_result_to_filename = "result_b2224.csv"

    model = efficientnet_classifierB2(input_shape=(224, 224, 3), num_of_cls=8)
    model.summary()
    model.load_weights(model_path)

    seq = iaa.Sequential([
        iaa.PadToAspectRatio(1.0, position='center', pad_cval=224),
        iaa.Resize({"height": 224, "width": 224})])

    result_fp = open(save_result_to_filename, "w")
    result_fp.write("ID_img,class\n")

    with open(test_list_filename, "r") as fp:
        for line in fp.readlines():
            label_data = line.replace("\n", "").split(",")
            img_name = label_data[0]
            if img_name.count(".jpg") == 0:
                continue

            image = cv2.imread(os.path.join(data_path, img_name))
            seq_det = seq.to_deterministic()
            augmented_image = seq_det.augment_images([image])
            image = augmented_image[0] / 255.

            model_pred = model.predict(np.expand_dims(image, axis=0))[0]

            res_str = "{},{}\n".format(img_name, np.argmax(model_pred))
            print(res_str)
            result_fp.write(res_str)

    result_fp.close()
