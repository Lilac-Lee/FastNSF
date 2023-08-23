### Pre-processing Waymo Open Scene Flow dataset

1. Download Waymo Open dataset from the [official website](https://waymo.com/open/download)

2. Unzip the downloaded ```.tar``` files to the directory of ```dataset/waymo/raw_data```. And make the dataset file structure like this:

    ```
    . dataset
    |-- waymo
    |   |-- ImageSets
    |   |   |-- train.txt
    |   |   |-- val.txt
    |   |-- raw_data
    |   |   |-- segment-xxxxxxx.tfrecord
    |   |   |-- ...
    ```

3. Install ```waymo-open-dataset```.
    ```
    sudo apt install npm
    sudo npm install -g @bazel/bazelisk
    git clone https://github.com/waymo-research/waymo-open-dataset.git
    cd waymo-open-dataset/src
    bazelisk build //waymo_open_dataset/pip_pkg_scripts:wheel
    cd bazel-bin/waymo_open_dataset/pip_pkg_scripts
    pip install waymo_open_dataset_tf_*-py3-none-any.whl
    ```

4. Run the following code in the [utils](./) folder.
    ```
    python generate_waymo_data.py --data_split val --root_data_dir ../dataset/waymo
    ```

5. The completed dataset file structure will be like this:

    ```
    . dataset
    |-- waymo
    |   |-- ImageSets
    |   |   |-- train.txt
    |   |   |-- val.txt
    |   |-- raw_data
    |   |   |-- segment-xxxxxxx.tfrecord
    |   |   |-- ...
    |   |-- processed
    |   |   |-- segment-xxxxxxx
    |   |   |-- ...
    |   |-- scene_flow
    |   |   |-- segment-xxxxxxx
    |   |   |   |-- 0000_0001.npz
    |   |   |   |-- ...
    |   |   |-- ...
    ```

6. Please note the above steps will generate the complete Waymo Open validation scene flow dataset. For this project, we only select the first scene flow data from each validation log files, which yields 202 examples in total.


### Acknowledgement

The dataset preprocessing steps were heavily borrowed from these projects:
[FastFlow3D](https://github.com/Jabb0/FastFlow3D), 
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet), 
[ST3D](https://github.com/CVMI-Lab/ST3D),
[DCA-SRSFE](https://github.com/leolyj/DCA-SRSFE).
