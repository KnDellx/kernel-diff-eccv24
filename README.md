# Kernel Diff: An Alternate Approach to Blind Deconvolution
Official PyTorch code and pre-trained models for [Kernel-Diff](https://arxiv.org/abs/2312.02319), accepted at ECCV 2024

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2312.02319)
[![poster](https://img.shields.io/badge/ECCV-Poster-blue)]([https://arxiv.org/abs/2312.02319](https://docs.google.com/presentation/d/1ovD1xFyPef3UHM1f7PwxL9AI7G9QyFmV/edit?usp=drive_link&ouid=113231491133219633899&rtpof=true&sd=true))
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://docs.google.com/presentation/d/1nsJwcD2a4CowkTgik2xMyAwIm8HgDcy9/edit?usp=sharing&ouid=113231491133219633899&rtpof=true&sd=true)

<center><img src="https://github.com/user-attachments/assets/5688788d-b205-4095-9782-9b497f004c3a"  width="800">
</center>

### Abstract
Blind deconvolution problems are severely ill-posed because neither the underlying signal nor the forward operator are not known exactly. Conventionally, these problems are solved by alternating between estimation of the image and kernel while keeping the other fixed. In this paper, we show that this framework is flawed because of its tendency to get trapped in local minima and, instead, suggest the use of a kernel estimation strategy with a non-blind solver. This framework is employed by a diffusion method which is trained to sample the blur kernel from the conditional distribution with guidance from a pre-trained non-blind solver. The proposed diffusion method leads to state-of-the-art results on both synthetic and real blur datasets.


### Pretrained Models here
Download the following pretrained models to reproduce ther results from the paper here <br>
[Kernel-Diff](https://drive.google.com/file/d/1yYonjCVMh6g-yYJISnY-WfgfPA4XXx43/view?usp=sharing) [Non-Blind Solver, DWDN](https://gitlab.mpi-klsb.mpg.de/jdong/dwdn/-/blob/master/model/model_DWDN.pt)

Pretrained model file for realistic blur kernels, used for results on the RealBlur dataset in the supplementary material <br>
[Kernel-Diff File](https://drive.google.com/file/d/1f7qxOcp6ubFVVRSaPLATj3eLPEb46Dc_/view?usp=sharing)



