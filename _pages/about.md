---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I am Jiaming Xu (许珈铭), a third year Ph.D student supervised by [Prof. Guohao Dai (戴国浩)](https://dai.sjtu.edu.cn/pepledetail.html?id=218) in School of Computer Science, Shanghai Jiao Tong University ([上海交通大学计算机学院](https://www.cs.sjtu.edu.cn/)) and Shanghai Innovation Institute ([上海创智学院](https://www.sii.edu.cn/)). Previously, I obtained my Bachelor's degree in 2023 from School of Computer Science and Technology, Xidian University ([西安电子科技大学计算机科学与技术学院](https://cs.xidian.edu.cn/)) supervised by [Prof. Nannan Wang (王楠楠)](https://web.xidian.edu.cn/nnwang/). I was once an intern in Infinigence AI ([无问芯穹](https://cloud.infini-ai.com/platform/ai)) and now still collaborate closely with Xiuhong Li ([李秀红](https://www.zhihu.com/people/hong_pku)) in Infinigence AI.

My research focuses on efficient machine learning systems (MLSys), primarily the effcient AI (e.g., LLM, sparse computing, embodied AI, multimodal model) inference through algorithm (e.g., quantization, pruning, speculative decoding) and system (kernel design, memory management, dataflow design, heterogeneous computing) co-deisgn. I have published 10+ papers <a href='https://scholar.google.com/citations?user=DhtAFkwAAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a> at the top international conferences and journals such as IEEE TCAD, ISCA, MLSys, DAC.


# 🔥 News
- *2025.11*: &nbsp;🎉🎉 [SpecDiff](https://www.arxiv.org/abs/2509.13848) and MoSs was accepted by AAAI! <span style="color: red;">SpecDiff is accepted as Oral presentation!</span>
- *2025.10*: &nbsp;🎉🎉 [MARCA-v2](https://ieeexplore.ieee.org/document/11214419) was accepted by IEEE TCAD!
- *2025.10*: &nbsp;🎉🎉 Congratulation! I was awarded the National Scholarship (Ph.D Students). This is my fourth National Scholarship.
- *2025.09*: &nbsp;🎉🎉 We release [SpecDiff](https://www.arxiv.org/abs/2509.13848) and [SpecPrune-VLA](https://arxiv.org/abs/2509.05614) for diffusion and VLA model acceleration.
- *2025.09*: &nbsp;🎉🎉 Two papers were accepted by ASP-DAC 2026. Look forward to meeting you next January in Hong Kong!
- *2025.08*: &nbsp;🎤🎤 A invited talk (大模型推理软硬件协同优化) was given in Ordos, China. Thanks for the invitation of CCF-HPC 2025. 
<!-- - *2025.06*: &nbsp;🎤🎤 Oral presentation in ISCA 2025 in Tokyo, Japan. -->
<!-- - *2025.03*: &nbsp;🎉🎉 A paper was accpeted by ISCA 2025. Look forward to meeting you in June in Tokyo! -->


# 👥 Team
Now I lead the system team (DAI-Sys) in our lab. Our team currently consists of 9 students, including 3 Ph.D. students, 1 master student, and 5 undergraduates. I am very happpy to cooperate with them. <span style="color: red;">I am looking for students, who are excited to tackle efficiency problems in AI from an algorithm, modeling, system/hardware perspectives, to join us</span> ([DAI-Sys小组招生](https://dai.sjtu.edu.cn/join.html)).

**Now**
- Jiaming Xu (许珈铭): third year Ph.D student in Shanghai Jiao Tong University and Shanghai Innovation Institude
- Yaoxiu Lian (廉瑶秀): fourth year Ph.D student in Shanghai Jiao Tong University
- Yongkang Zhou (周永康): first year Ph.D student in Shanghai Jiao Tong University and Shanghai Innovation Institude
<!-- - Kele Shao (邵可乐): first year Ph.D student in Westlake University and Shanghai Innovation Institude -->
- Jiayi Pan (潘佳一): second year master student in Shanghai Jiao Tong University
<!-- - Tianlang Zhao (赵天朗): fourth year undergraduate in Shanghai Jiao Tong University -->
- Hanzhen Wang (王翰楨): third year undergraduate in Shanghai Jiao Tong University
- Yifan Jiao (焦一帆): second year undergraduate in Shanghai Jiao Tong University
- Mingyi Xu (徐铭怿): second year undergraduate in Shanghai Jiao Tong University
- Chengze Yuan (袁诚泽): second year undergraduate in Shanghai Jiao Tong University
- Haotian Fang (方皓天): second year undergraduate in Shanghai Jiao Tong University



**Previous**
- Siming Chen (陈思铭, 2024.10~2025.6): fourth year undergraduate in Lanzhou University
- Junyi Wu (吴俊逸, 2024.1~2025.3): third year undergraduate in Shanghai Jiao Tong University



# 📝 Publications 

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ISCA 2025</div><img src='images/ISCA25.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting](https://dai.sjtu.edu.cn/my_file/pdf/7e067065-0e58-4e87-a373-feea0bebde1b.pdf)

**Jiaming Xu**, Jiayi Pan, Yongkang Zhou, Siming Chen, Jinhao Li, Yaoxiu Lian, Junyi Wu, Guohao Dai

<span style="color: red;">ISCA 2025 (CCF-A)</span>

[**Paper**](https://dai.sjtu.edu.cn/my_file/pdf/7e067065-0e58-4e87-a373-feea0bebde1b.pdf) <strong>|</strong> [**Code**](https://github.com/infinigence/SpecEE) ![](https://img.shields.io/github/stars/infinigence/SpecEE) <strong>|</strong>  [**机器之心**](https://mp.weixin.qq.com/s/vecJX1J8sFoRK8ZudFfzaA) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:WF5omc3nYNoC'></span></strong> 
</div>
</div>




<div class='paper-box'><div class='paper-box-image'><div><div class="badge">IEEE TCAD 2025</div><img src='images/TCAD25.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Enabling Efficient Sparse Multiplications on GPUs with Heuristic Adaptability](https://ieeexplore.ieee.org/document/10802949)

**Jiaming Xu\***, Shan Huang\*, Jinhao Li, Guyue Huang, Yuan Xie, Yu Wang, Guohao Dai

<span style="color: red;">IEEE TCAD 2025 (CCF-A)</span>

[**Paper**](https://ieeexplore.ieee.org/document/10802949) <strong>|</strong> [**Project**](https://dgsparse.github.io/) <strong>|</strong> [**Code**](https://github.com/dgSPARSE) ![](https://img.shields.io/github/stars/dgSPARSE/dgSPARSE-Lib) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:Tyk-4Ss8FVUC'></span></strong> 
</div>
</div>




<div class='paper-box'><div class='paper-box-image'><div><div class="badge">MLSys 2024</div><img src='images/MLSys24.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf)

Ke Hong\*, Guohao Dai\*, **Jiaming Xu\***, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong and Yu Wang

MLSys 2024 (non-CCF)

[**Paper**](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf) <strong>|</strong> [**Arxiv**](https://arxiv.org/abs/2311.01282) <strong>|</strong> [**机器之心**](https://mp.weixin.qq.com/s/e9OHATqk88Q9zM3Y5czFQA)  <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:qjMakFHDy7sC'></span></strong> 
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='images/SpecDiff.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](https://www.arxiv.org/abs/2509.13848)

Jiayi Pan*, **Jiaming Xu\***, Yongkang Zhou, Guohao Dai

<span style="color: red;">AAAI 2026 Oral (CCF-A)</span>

[**Arxiv**](https://www.arxiv.org/abs/2509.13848) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:5nxA0vEk-isC'></span></strong> 
</div>
</div>




<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='images/SpecVLA.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[SpecPrune-VLA: Accelerating VisionLanguage-Action Models via Action-Aware Self-Speculative Pruning](https://arxiv.org/abs/2509.05614)

Hanzhen Wang\*, **Jiaming Xu\***, Jiayi Pan, Yongkang Zhou, Guohao Dai

[**Arxiv**](https://arxiv.org/abs/2509.05614) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:0EnyYjriUFMC'></span></strong> 
</div>
</div>






<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='images/aspdac26.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

SpAct-NDP: Efficient LLM Inference via Sparse Activation on NDP-GPU Heterogeneous Architecture

ASP-DAC 2026 (CCF-C)

**Jiaming Xu\***, Tongxin Xie\*, Yongkang Zhou, Jinhao Li, Yaoxiu Lian, Zhenhua Zhu, Yu Wang, Guohao Dai


</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='images/aspdac25.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Accelerator for LLM-Enhanced GNN with Product Quantization and Unified Indexing](https://dai.sjtu.edu.cn/my_file/pdf/8543405a-e6d3-48de-89b9-f1a89e0a4ae9.pdf)

**Jiaming Xu\***, Jinhao Li\*, Jun Liu, Hao Zhou and Guohao Dai 

ASP-DAC 2025 (CCF-C)

[**Paper**](https://dai.sjtu.edu.cn/my_file/pdf/8543405a-e6d3-48de-89b9-f1a89e0a4ae9.pdf) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:YsMSGLbcyi4C'></span></strong> 
</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICCAD 2024</div><img src='images/iccad24.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Fast and Efficient 2-bit LLM Inference on GPU: 2/4/16-bit in a Weight Matrix with Asynchronous Dequantization](https://dai.sjtu.edu.cn/my_file/pdf/f1c6a4bb-4556-43e2-8e46-4ab38d8bfed4.pdf)

Jinhao Li\*, **Jiaming Xu\***, Shiyao Li, Shan Huang, Jun Liu, Yaoxiu Lian and Guohao Dai

ICCAD 2024 (CCF-B)

[**Paper**](https://dai.sjtu.edu.cn/my_file/pdf/f1c6a4bb-4556-43e2-8e46-4ab38d8bfed4.pdf) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:eQOLeE2rZwMC'></span></strong> 
</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='images/hardware_survey.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective](https://arxiv.org/abs/2410.04466)

Jinhao Li, **Jiaming Xu**, Shan Huang, Yonghua Chen, Wen Li, Jun Liu, Yaoxiu Lian, Jiayi Pan, Li Ding, Hao Zhou, Yu Wang, Guohao Dai

[**Arxiv**](https://arxiv.org/abs/2410.04466) <strong>|</strong> [**Code**](https://github.com/Kimho666/LLM_Hardware_Survey) <strong>|</strong> [**推广**](https://mp.weixin.qq.com/s/_rvKIZB1IzsX7Z6oe1a2rw) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:UeHWp8X0CEIC'></span></strong> 
</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='images/tsinghua_survey.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)

Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, **Jiaming Xu**, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Yuhan Dong, Yu Wang

[**Arxiv**](https://arxiv.org/abs/2404.14294) <strong>|</strong> [**机器之心**](https://mp.weixin.qq.com/s/7LKfamTnCyFih6_grf9m3A) <strong><span class='show_paper_citations' data='ovQFaicAAAAJ:2osOgNQ5qMEC'></span></strong> 
</div>
</div>




- `IEEE TCAD 2026` <span style="color: red;">(CCF-A)</span>, [MARCA-v2: Mamba Accelerator with Complementary State Space Model Sparsity and Reconfigurable Architecture](https://ieeexplore.ieee.org/document/11214419), Jinhao Li\*, Shan Huang\*, **Jiaming Xu**, Jun Liu, Ningyi Xu, Guohao Dai


- `IEEE TC 2025` <span style="color: red;">(CCF-A)</span>, [FlashDecoding++Next: High Throughput LLM Inference with Latency and Memory Optimization](https://ieeexplore.ieee.org/document/11062854/), Guohao Dai, Ke Hong, Qiuli Mao, Xiuhong Li, **Jiaming Xu**, Haofeng Huang, Hongtu Xia, Xuefei Ning, ShengenYan, Yun Liang, Yu Wang

- `AAAI 2026` <span style="color: red;">(CCF-A)</span>, [MoSs: Mixture of Scales for Efficient High-Resolution Autoregressive Image Generation](https://openreview.net/pdf?id=sdm8iVl0A3), Yaoxiu Lian, Liang Hao, Gou Zhihong, Yijia Zhang, **Jiaming Xu**, Guohao Dai, Ningyi Xu 


- `ASP-DAC 2026`(CCF-C), BalanceGS: AlgorithmSystem Co-design for Efficient 3D Gaussian Splatting Training on GPU, Junyi Wu*, **Jiaming Xu\***, Jinhao Li, Yongkang Zhou, Jiayi Pan, Xingyang Li, Guohao Dai

- `CIKM 2025` (CCF-B), SG-Filter: Enhancing Similar Text Retrieval via Hierarchical Summarized-Semantic Index and Adaptive Filtering, Jiancai Ye\*, Jun Liu\*, Haoyu Zhang, Maojia Sheng, Tao Yang, **Jiaming Xu**, Jinhao Li, Yu Wang, Guohao Dai

- `DAC 2025` <span style="color: red;">(CCF-A)</span>, [A Cross-model Fusion-aware Framework for Optimizing (gather-matmul-scatter)s Workload](https://dai.sjtu.edu.cn/my_file/pdf/a5764a0c-bde6-46df-b884-92e89c8e9cbf.pdf), Yaoxiu Lian, Zhihong Gou, Yibo Han, Zhongming Yu, **Jiaming Xu**, Sheng Yuan, Zhilin Pei, Xingcheng Zhang, Ningyi Xu and Guohao Dai

- `DATE 2025` (CCF-B), [DyLGNN: Efficient LM-GNN Fine-tuning with Dynamic Node Partitioning, Low-degree Sparsity, and Asynchronous Sub-batch](https://dai.sjtu.edu.cn/my_file/pdf/08856e49-c2eb-4421-b5cd-ebd4d436fb5c.pdf), Zhen Yu\*, Jinhao Li\*, **Jiaming Xu**, Shan Huang, Jiancai Ye, Ningyi Xu and Guohao Dai

- `ASP-DAC 2025`(CCF-C), [LLSM: LLM-enhanced Logic Synthesis Model with EDA-guided CoT Prompting, Hybrid Embedding and* AIG-tailored Acceleration](https://dai.sjtu.edu.cn/my_file/pdf/739d9f50-3749-47e0-a40f-3c5a63c54b31.pdf), Shan Huang\*, Jinhao Li\*, Zhen Yu, Jiancai Ye, **Jiaming Xu**, Ningyi Xu and Guohao Dai


- `ICCAD 2024` (CCF-B), [MARCA: Mamba Accelerator with Reconfigurable Architecture](https://dai.sjtu.edu.cn/my_file/pdf/010527f5-f0b2-4859-8c27-9052bd14da49.pdf), Jinhao Li\*, Shan Huang\*, **Jiaming Xu**, Jun Liu, Li Ding, Ningyi Xu and Guohao Dai

- `ICCAD 2023` (CCF-B), [TSTC: Two-level Sparsity Tensor Core Enabling both Algorithm Flexibility and Hardware Efficiency](https://dai.sjtu.edu.cn/my_file/pdf/9b9ce3a9-703e-48a7-88be-9b9330a34dfe.pdf), Jun Liu, Guohao Dai, Hao Xia, Lidong Guo, Xiangsheng Shi, **Jiaming Xu**, Huazhong Yang and Yu Wang










<!-- - [Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ornare aliquet ipsum, ac tempus justo dapibus sit amet](https://github.com), A, B, C, **CVPR 2020** -->

# 🎖 Honors and Awards
- *2025.10* <span style="color: red;">National Scholarship</span> (Top 1%), Ministry of Education of The People’s Republic of China.
- *2023.06* Outstanding Graduates, Shaanxi.
- *2023.06* Outstanding Graduates, Xidian University.
- *2023.06* Graduate Star (1/10), Xidian University.
- *2022.12* Thanks for the Modern Scientist Scholarship (感恩近现代科学家奖学金) (1/12), Xidian University.
- *2022.12* Principal’s Scholarship (校长奖学金) (1/5), Xidian University.
- *2022.12* <span style="color: red;">National Scholarship</span> (Top 1%), Ministry of Education of The People’s Republic of China.
- *2022.10* Top 1.3% of World, IEEEXtreme Programming Competition.
- *2022.10* Huawei Intelligent Base Scholarship (华为智能基座奖学金) (1/10), Xidian University.
- *2021.12* <span style="color: red;">National Scholarship</span> (Top 1%), Ministry of Education of The People’s Republic of China.
- *2021.11* Silver Medal, The ICPC International Collegiate Programming Contest of Shaanxi Province.
- *2021.10* Huawei Intelligent Base Scholarship (华为智能基座奖学金) (1/10), Xidian University.
- *2020.12* <span style="color: red;">National Scholarship</span> (Top 1%), Ministry of Education of The People’s Republic of China.
- *2020.11* Bronze Medal, The ICPC International Collegiate Programming Contest of Shaanxi Province.


# 📖 Educations
- *2023.06 - 2028.06 (expected)*, School of Computer Science, Shanghai Jiao Tong University 
- *2019.09 - 2023.06*, School of Computer Science and Technology, Xidian University

# 💬 Presentation
- *2025.08*, **\[Invited Talk\]** Efficient LLM inference via hardware-software codesign, CCF-HPC 2025 @Ordos, China
- *2025.06*, **\[Oral Presentation\]** Accelerating Large Language Model Inference with Speculative Early Exiting, ISCA 2025 @Tokyo, Japan
- *2024.12*, **\[Oral Presentation\]** Efficient LLM Inference on GPUs with Operator Optimization and Compilation, Chinasys 2024 @Tianjin, China
- *2024.11*, **\[Oral Presentation\]** MARCA: Mamba Accelerator with Reconfigurable Architecture, ICCAD 2024 @New York, USA
- *2024.11*, **\[Oral Presentation\]** Fast and Efficient 2-bit LLM Inference on GPU: 2/4/16-bit in a Weight Matrix with Asynchronous Dequantization, ICCAD 2024 @New York, USA
- *2024.11*, **\[Oral Presentation\]** Towards Floating Point-Based Attention-Free LLM: Hybrid PIM with Non-Uniform Data Format and Reduced Multiolications, ICCAD 2024 @New York, USA
- *2024.09*, **\[Invited Talk\]** Efficient GPU computation in Large Language Models, CCF-HPC 2024 @Wuhan, China
- *2024.05*, **\[Oral Presentation\]** FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics, MLSys 2024 @California, USA
- *2024.03*, **\[Invited Talk\]** NVIDIA GPU and LLM Exploration, Flat GEMM Optimization, SJTU @Shanghai, China

# 💻 Service
- *2025.09 - Now*, Teaching Assistant, Thinking and Methodology in Programming (C++) (UG-CS1501-08) by [Prof. Weiguo Gu](https://me.sjtu.edu.cn/teacher_directory1/guweiguo.html)
- *2025.02 - 2025.06*, Teaching Assistant, Algorithms and Complexity (UG-CS2308-01) by [Prof. Qingshen Ren](https://www.cs.sjtu.edu.cn/PeopleDetail.aspx?id=85)
- *2025.02 - 2025.06*, Teaching Assistant, Algorithm Design and Analysis (PG-CS7310H-033-M01) by [Prof. Guohao Dai](https://dai.sjtu.edu.cn/pepledetail.html?id=218)
- *2024.09 - 2025.01*, Teaching Assistant, Thinking and Methodology in Programming (C++) (UG-CS1501-04) by [Prof. Weiguo Gu](https://me.sjtu.edu.cn/teacher_directory1/guweiguo.html)
- *2024.05 - Now*, IT Administrator of DAI-Lab, Shanghai Jiao Tong University
- *2025.02 - 2025.06*, Teaching Assistant, Algorithm Design and Analysis (PG-CS7310H-033-M01) by [Prof. Guohao Dai](https://dai.sjtu.edu.cn/pepledetail.html?id=218)
- *2022.09 - 2023.06*, Huawei Campus Ambassador, Xidian University
- *2021.09 - 2022.09*, Chairman of Huawei Innovation Club and Huawei Intelligent Base Club
