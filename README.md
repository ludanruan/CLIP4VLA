# CLIP4VLA
The official code base of Accommodating Audio Modality in CLIP for Multimodal Processing [CIP4VLA](https://arxiv.org/pdf/2303.06591.pdf)
## Setup
```
conda create -n clip4vla python=3.7
conda activate clip4vla
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tqdm transformers soundfile opencv-python boto3 ftfy pandas
pip install h5py librosa dominate
```
## Preparation
#### Dataset
Download MSR-VTT from [Baiduyun](https://pan.baidu.com/s/11VWH8VqczIj42LXJ3Y-wkA?pwd=qhq7) (passward:qhq7)
unzip it with```tar -zxvf msrvtt.tar.gz``` and place it in `./data`.
process the dataset with the following command:
```
python data_processor.py --extract_audios --load_video_into_frames
cd data/msrvtt
mv softlink.sh audios_16k/
cd audios_16k
bash softlink.sh
``` 

## Pre-train
To pretrain from scratch, first prepare the dataset of Howto100M and Audioset. Then run the following command:
```
bash ./scripts/pretrain_howto100m_s1.sh
```

## Fine-tune
Prepare the dataset of MSR-VTT or vatex with ```data_processor.py``` and then run the following command:
```
bash ./scripts/<dataset_task>/finetune_retrieval_vatex_pre_video.sh
```





