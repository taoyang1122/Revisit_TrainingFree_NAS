# Revisiting Training-free NAS Metrics: An Efficient Training-based Method

This repo is the implementation of [Revisiting Training-free NAS Metrics: An Efficient Training-based Method](https://arxiv.org/abs/2211.08666) at WACV 2023.

If you find our work useful in your research, please consider citing us:

```bibtex
@inproceedings{yang2023revisiting,
  title={Revisiting Training-free NAS Metrics: An Efficient Training-based Method},
  author={Yang, Taojiannan and Yang, Linjie and Jin, Xiaojie and Chen, Chen},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4751--4760},
  year={2023}
}
```

## Usage 

python = 3.8.12, pytorch = 1.7.1

Install NASBench following instructions in https://github.com/google-research/nasbench

Download the NASbench101 data (see https://github.com/google-research/nasbench)

Download the NASbench201 data (see https://github.com/D-X-Y/NAS-Bench-201)

To run the experiments, use the script:

`bash run_exp.sh`

## Acknowledgement

The code is based on the implementation of [NASWOT](https://github.com/BayesWatch/nas-without-training)
