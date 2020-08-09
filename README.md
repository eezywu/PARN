# Position-Aware Relation Networks
The pytorch implementation of Position-Aware Relation Networks (PARN), which is proposed in [Position-Aware Relation Networks for Few-Shot Learning](https://arxiv.org/abs/1909.04332).

## Usage
- Build environments of Deformable Convolutional Networks (DCN)
    - Download the [DCN code](https://github.com/CharlesShang/DCNv2), and follow the guideline to build environments of DCN. If you fail to complie the above code, you can try [this code](https://github.com/jinfagang/DCNv2_latest).
- Download the Mini-Imagenet dataset, and put it in the `datas/miniImagenet` folder. You can follow the steps of [this code](https://github.com/floodsung/LearningToCompare_FSL) to prepare the dataset.
- Run `pip install -r requirements.txt`
- Run `python main.py`

## Citation
```
@article{wu2019parn,
    title={PARN: Position-Aware Relation Networks for Few-Shot Learning},
    author={Wu, Ziyang and Li, Yuwei and Guo, Lihua and Jia, Kui},
    journal={ICCV},
    year={2019}
}
```
