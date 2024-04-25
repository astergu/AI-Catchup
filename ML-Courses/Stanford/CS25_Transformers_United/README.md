# CS25: Transformers United V4
 Since their introduction in 2017, Transformers have revolutionized Natural Language Processing (NLP). Now, Transformers are finding applications all over Deep Learning, be it Computer Vision (CV), Reinforcement Learning (RL), Generative Adversarial Networks (GANs), Speech, or even Biology. Among other things, Transformers have enabled the creation of powerful language models like ChatGPT and GPT-4 and significantly elevated the capabilities and impact of artificial intelligence.

In this seminar, we examine the details of how Transformers work, and dive deep into the different kinds of Transformers and how they're applied in various fields, with a focus on LLMs. We do this through a combination of instructor lectures, guest lectures, and classroom discussions. We will invite people at the forefront of Transformers research across different domains for guest lectures. 

[[course page]](https://web.stanford.edu/class/cs25/)

**Prerequisites**
- Basic knowledge of Deep Learning (should understand attention)
- Or CS224N/CS231N/CS230. 

**Previous Courses**
- **Transformers United V1**
  - [[homepage]](https://web.stanford.edu/class/cs25/prev_years/2021_fall/index.html)
  - [[video]](https://www.youtube.com/watch?v=P127jhj-8-Y&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM)
- **Transformers United V2**
  - [[homepage]](https://web.stanford.edu/class/cs25/prev_years/2023_winter/index.html)
  - [[video]](https://www.youtube.com/watch?v=XfpMkf4rD6E&list=PLVVTN-yNn8rvEwlY8ClxDUWeVPVfdifYj)
- **Transformers United V3**
  - [[homepage]](https://web.stanford.edu/class/cs25/)
  - [[video]](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) (all of V1, V2, V3)


## Lecture Notes

> Since [V2 page](https://web.stanford.edu/class/cs25/prev_years/2023_winter/index.html) has the most detailed schedule and corresponding course materials, I will update the lecture notes mainly according to the V2 page.<br>
> Lecture notes are taken regarding [V1/V2/V3 video playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM). <br><br>

> [!IMPORTANT]
> Since April 23, 2024, the first V4 video has been uploaded! We will take notes for V4 videos if they keep uploading new ones!

 
> Version representations:
> - :white_check_mark:  V1
> - :large_blue_circle:  V2
> - :large_orange_diamond: V3
> - :red_circle:  V4

| Lecture | Course Materials | Key Takeaways |
| ---- | ---- | ---- |
| :red_circle: [Overview of Transformers](https://www.youtube.com/watch?v=fKMB5UlVY1E) | Slides: <br> :red_circle: [Overview of Transformers](./slides/week1/CS25%20V4%20Lecture%201%20(Spring%202024).pdf) | [[Detailed Notes]](#lecture-1-overview-of-transformers) |
| :red_circle: []() | :red_circle: [Intuitions on Language Models](./slides/week2/2024%20stanford%20cs25%20guest%20lecture%20jason%20wei.pdf) <br> :red_circle: [Shaping the Future of AI from the History of Transformer](./slides/week2/Stanford_CS_25.pdf)  | |

### Lecture 1: Overview of Transformers
 
- Recent Trends of and Remaining Weakness of LLMs
  - BabyLM: Children vs. LLMs
    - Require large amounts of data, compute and cost
  - Minified LLMs and On-Device LLMs
    - AutoGPT and ChatGPT plugins
    - smaller open-source models (e.g., LLaMA, Mistral)
  - Memory Augmentation & Personalization
    - Weakness of LLMs is that they are frozen in knowledge at a particular point  in time, and don't augment knowledge "on the fly"
    - Potential approaches
      - Memory bank
      - Prefix-tuning approaches
      - Some prompt-based approach
      - RAG (retrieval-augmented generation)
  - Pretraining Data Synthesis & Selection
    - Example: Microsoft Phi models ("Textbooks Are All You Need")
    - Microsoft Phi-2: 2.7 billion-parameter model
  - New Knowledge or "Memorizing"?
  - Continual Learning
    - AKA, infinite and permanent fundamental self-improvement
  - Interpretability of LLMs
  - Model Modularity + Mixture of Experts (MoE)
  - Self-Improvement/Self-Reflection
  - Hallucination Problem
- Chain-of-thought (CoT)
  - Series of intermediate reasoning steps
  - Currently, CoT works effectively for models of approx. 100B params or more
  - Tree of Thoughts
    - consider multiple different reasoning paths and self-evaluating choices to decide the next course of action
  -  Socrative Questioning
     -  Divide-and-conquer fashing algorithm that simulates the self-questioning and recursive thinking process
-  From Language Models to AI Agents

### Lecture 2

