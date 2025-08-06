<div align="center">

# SSRL

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.16084)  [![Github](https://img.shields.io/badge/SSRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/YuchenFan48/SSRL)
[![Wandb Log of AIME](https://img.shields.io/badge/Wandb%20Log%20of%20AIME-%2300B4AB?style=for-the-badge&logo=weightsandbiases&logoColor=white&labelColor=000000)](https://wandb.ai/yuchenfan/Rethink%20Search%20Scaling/reports/SSRL--VmlldzoxMzg3Nzc0NA)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a> •
    <a href="#main-results" style="text-decoration: none; font-weight: bold;">📊 Main Results</a>
  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📨 Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#star-history" style="text-decoration: none; font-weight: bold;">🌟 Star History</a>
  </p>
</div>

> Welcome to the Era of Experience.  --David Silver, Richard S. Sutton

<!-- # 🎉News

- **[2025-05-23]** We update both the paper and the code, with the implementation based on the [verl](https://github.com/volcengine/verl).

- **[2025-04-24]** We release the code and experimental logs. Check it out: [Getting Started](#getting-started).

- **[2025-04-23]** We present **TTRL** (Test-Time Reinforcement Learning), an open-source solution for online RL on data without ground-truth labels, especially test data. -->

# 📖Introduction

**We investigate Reinforcement Learning (RL) on Agentic search tasks without explicit gathering information from external search engines, e.g., LLMs, web engines.**
Previous work leverage external search engines during training, which is expensive and time-consuming, yet introducing training instability. We introduce SSRL, a novel approach that enables RL on Agentic search tasks without the need for explicit search engines which achieves comparable performance to previous methods. Though trained totally offline, it can be seamlessly applied to online search engines, and further boost its performance. 

<p align="center">
   <img src="figs/teaser.pdf" alt="Performance and settings of SSRL." style="width: 80%;">
</p>

<!-- 
<p align="center">
   <img src="figs/overview.png" alt="Overview of TTRL." style="width: 80%;">
</p> -->


# 📊Main Results

Our experiments demonstrate that SSRL consistently improves performance across a variety of tasks and models. 

Furthermore, although SSRL is trained offline, it can be seamlessly applied to online search engines, further boosting its performance.

<p align="center">
   <img src="figs/results.png" alt="Main results of SSRL." style="width: 60%;">
</p>

<p align="center">
   <img src="figs/results_sim.png" alt="Main results of SSRL." style="width: 60%;">
</p>


# ✨Getting Started

You can reproduce the results of SSRL with the following commands:

```bash
git clone https://github.com/YuchenFan48/SSRL
cd verl

pip install -r requirements.txt

huggingface-cli download --repo-type dataset --resume-download yuchenFan/SSRL --local-dir SSRL_dataset # download the dataset

bash examples/ssrl/example.sh
```

*All experiments were conducted on 8 x NVIDIA A800 80GB GPUs.*

# 📨Contact

- Kaiyan Zhang: zhang-ky22@mails.tsinghua.edu.cn

- Ning Ding: dingning@mail.tsinghua.edu.cn

# 🎈Citation
If you find SSRL helpful, please cite us.

```bibtex
<!-- @article{zuo2025ttrl,
  title={Ttrl: Test-time reinforcement learning},
  author={Zuo, Yuxin and Zhang, Kaiyan and Qu, Shang and Sheng, Li and Zhu, Xuekai and Qi, Biqing and Sun, Youbang and Cui, Ganqu and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2504.16084},
  year={2025}
} -->
```

# 🌟Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PRIME-RL/TTRL&type=Date)](https://www.star-history.com/#PRIME-RL/TTRL&Date)