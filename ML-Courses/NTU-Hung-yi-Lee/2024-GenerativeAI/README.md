- [About the Class](#about-the-class)
- [Syllabus](#syllabus)
- [Key Takeaways](#key-takeaways)
  - [Lecture 1：课程概述](#lecture-1课程概述)
    - [80分钟快速了解大型语言模型](#80分钟快速了解大型语言模型)
- [References](#references)


# About the Class

- 这门课不是教你如何使用ChatGPT，而是帮助你了解ChatGPT之类生成式AI背后的原理及未来发展和可能性。
- 不需要预修其他课程。
- 可以作为学习人工智能的第一门课，因此无法深入提到所有技术，如果想要更深入研究，推荐阅读PPT中引用的论文。（上课引用论文多数来自[Self-Supervised Speech Representation Learning: A Review](https://arxiv.org/abs/2205.10643)）
- 作业难度分为三个等级，分别是：
  - :white_check_mark: (送分) 
  - :heavy_exclamation_mark:（如果不在意分数可以在短时间内完成）
  - :bangbang:（至少需要数小时）


# Syllabus


| Date | Topic | Slides |  Extra Materials | Presentation | Homework |
| -------- | ------- | -------- | ------- | -------- | -------- |
| 02/23 | **课程概述** <br> [[课程说明]](https://youtu.be/AVIKFXLCPY8) <br> [[课程规则]](https://youtu.be/vCxyd_S4R24)  <br> [[第1讲：生成式AI是什么？]](https://youtu.be/JGtqpQXfJis) |  1. [[课程内容说明]](./slides/01/0223_course.pdf) <br> 2. [[什么是生成式人工智能？]](./slides/01/0223_universal.pdf) <br> 3. [[今日的生成式人工智能厉害在哪里]](./slides/01/0223_intro_gai.pdf) | [80分钟快速了解大型语言模型](https://youtu.be/wG8-IUtqu-s?si=-YWWLqbeX7wiRQ4M) <br> [[slide]](./slides/01/LLM_80min%20(v5).pdf) |  | 真假难辨的世界 :white_check_mark: <br> [[video]](https://www.youtube.com/watch?v=QOrtPUxaIG8) [[pdf]](./assignments/HW1/GenAI%20HW1%20slides.pdf) |
| 03/01 | 提示工程&AI代理人 |  | | | 都是AI的作文比赛 :white_check_mark: | 
| 03/08 | 生成策略&从专才到通才 | | | | 以AI搭建自己的应用 :heavy_exclamation_mark: | 
| 03/15 | 生成策略&从专才到通才 | ||  MTK团队演讲 |  |
| 03/22 | 深度学习&Transformer | | |  | 成为AI催眠大师 :heavy_exclamation_mark:  |
| 03/29 | 深度学习&Transformer | | | NVIDIA团队演讲 | 训练自己的语言模型 :bangbang: |
| 04/12 | 评估生成式AI&道德议题 | | | | AI透过人类的回馈学习 :heavy_exclamation_mark: | 
| 04/26 | 打造生成式AI经验谈 | | | TAIDE团队演讲 | |
| 05/03 | 生成式AI的可解释性 | | | | 了解生成式AI在想什么 :heavy_exclamation_mark: |
| 05/10 | 语音的生成式AI  | | | | 生成式AI的安全性议题 :heavy_exclamation_mark: |
| 05/17 | 语音的生成式AI | | | | 演讲影片快速摘要 :heavy_exclamation_mark: |
| 05/24 | 影像的生成式AI | | | MTK团队演讲 | |
| 05/31 | 影像的生成式AI | | | | 定制化自己的影像生成AI :bangbang: |

# Key Takeaways

## Lecture 1：课程概述

- `生成式AI`：是人工智能的一个子集，目标是使得机器产生复杂有结构的物件（能够产生在训练时从来没有看过的东西），比如，写作一篇文章，画一幅画。如今的生成式AI通常采用**深度学习**（e.g. Transformer）来完成任务。
- `机器学习`：机器自动从资料中找到一个函数来拟合。
- `生成式问答`：生成式AI的案例之一。把任务拆解成一连串的文字接龙，一个字接一个字地产生，即`语言模型`。 

### 80分钟快速了解大型语言模型

- **ChatGPT**：由OpenAI开发的大型语言模型
  - `G`: Generative 生成
  - `P`: Pre-trained 预训练
  - `T`: Transformer
- ChatGPT的本质就是做`文字接龙`。
![ChatGPT真正做的事](./image/what_do_chatgpt_do.png)
- 为什么不每次都选机率最大的token作为接下去的词呢？因为效果不一定好，很可能会不断重复一样的内容 —— [[The Curious Case of Neural Text Degeneration]](https://arxiv.org/abs/1904.09751)
- 有预训练后，督导式学习不用大量资料。在多种语言上做预训练后，只要教某一个语言的某一个任务，自动学会其他语言的同样任务。
![GPT](image/gpt.png)
- 如何更好地使用ChatGPT？
  - 把需求讲清楚
  - 提供资讯给ChatGPT
  - 提供范例
  - 鼓励ChatGPT想一想
  - 如何找出神奇咒语
    - 通过强化学习找出有效prompt [[Learning to Generate Prompts for Dialogue Generation through Reinforcement Learning]](https://arxiv.org/abs/2206.03931)
    - [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
  - 可以上传资料，e.g.，图片，PPT等。
  - 调用ChatGPT插件
  - 帮助ChatGPT拆解任务
  - 引导ChatGPT自主进行规划，把任务拆解成小任务
  - ChatGPT其实是会反省的，e.g.，`请检查上述资讯是否正确`
  - 跟真实环境互动

# References

> 包括课程或讲座中提到的论文。

- 讲座：[AI時代，你跟上了嗎？｜李宏毅｜人文講堂](https://www.youtube.com/watch?v=iqaiPyvDD4Y)
- [Self-Supervised Speech Representation Learning: A Review](https://arxiv.org/abs/2205.10643)
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) 
- [Re3: Generating Longer Stories With Recursive Reprompting and Revision](https://arxiv.org/abs/2210.06774) 如何引导LLM来写长篇小说
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) 要求LLM自我反省其回答是否无害
- [DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents](https://arxiv.org/abs/2303.17071) 两个LLM彼此给对方的回答挑错，以此提升回答质量
- [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://arxiv.org/abs/2201.07207) 如何让LLM跟真实环境互动
- [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://innermonologue.github.io/)