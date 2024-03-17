- [GenAI HW2：都是AI的作文比赛](#genai-hw2都是ai的作文比赛)
  - [任务](#任务)
  - [Part 1: 用AI模型来生成文章](#part-1-用ai模型来生成文章)
    - [生成英文文章](#生成英文文章)
      - [回答一：原始prompt](#回答一原始prompt)
    - [生成中文文章](#生成中文文章)
      - [回答一：原始prompt](#回答一原始prompt-1)
    - [生成prompt示例](#生成prompt示例)
  - [Part 2: 用AI模型来评估文章](#part-2-用ai模型来评估文章)

# GenAI HW2：都是AI的作文比赛

## 任务

- 用AI模型来生成两篇文章：中文/English（也可以自己撰写）
- 用AI模型来评估两篇文章：中文/English

## Part 1: 用AI模型来生成文章

### 生成英文文章

> Do you agree or disagree with the following statement?
> Artificial Intelligence will eventually replace humans in most areas of work in the future world.
> Write an English essay with around 300 words to express your instance with specific reasons and examples to support your answer.

#### 回答一：原始prompt

> 以下文字由ChatGPT 3.5生成

As an AI language model, I don't have personal opinions or beliefs. However, I can provide arguments on both sides of the statement.

It is undeniable that artificial intelligence (AI) has made significant advancements and continues to evolve rapidly, leading to speculation about its potential to replace humans in various areas of work in the future world. While it is true that AI possesses capabilities such as automation, data analysis, and pattern recognition that can enhance efficiency and productivity, the idea that it will entirely replace humans in most areas of work is not entirely accurate.

Firstly, it's essential to acknowledge that AI is designed to complement human capabilities, not entirely supplant them. While AI can perform repetitive tasks with precision and speed, it lacks the nuanced understanding, creativity, and empathy that humans bring to many roles. For example, in fields like healthcare and education, human interaction and emotional intelligence are crucial components that cannot be replicated by AI alone. A compassionate bedside manner or the ability to inspire and motivate students are attributes unique to humans.

Moreover, there are certain tasks and professions where human judgment and intuition are irreplaceable. For instance, in complex decision-making processes, especially those involving ethical dilemmas or unpredictable scenarios, human cognition is indispensable. AI may offer suggestions based on data analysis, but the final decision often requires human oversight and moral reasoning.

Furthermore, the widespread adoption of AI in the workplace is not without challenges. Concerns about job displacement, ethical implications, and biases in AI algorithms are valid considerations that need to be addressed. Additionally, the implementation and maintenance of AI systems require human expertise, creating new job opportunities in AI development, maintenance, and oversight.

In conclusion, while AI undoubtedly has the potential to transform the future of work, it is unlikely to entirely replace humans in most areas. Instead, the optimal scenario is one where AI and humans collaborate synergistically, leveraging the strengths of each to achieve greater innovation, efficiency, and societal benefit.

### 生成中文文章

> 参考[113学年度学科能力测验试题：国语文写作能力测验 第二部分](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ceec.edu.tw/files/file_pool/1/0O021612121259980745/07-113%E5%AD%B8%E6%B8%AC%E5%9C%8B%E5%AF%AB%E8%A9%A6%E9%A1%8C.pdf)所附的文章，请以【缝隙的联想】为题，撰写一篇约五百字的繁体中文文章，并结合生活经验或见闻，书写你的感思与体悟。

- [113学年度学科能力测验试题：国语文写作能力测验 第二部分](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ceec.edu.tw/files/file_pool/1/0O021612121259980745/07-113%E5%AD%B8%E6%B8%AC%E5%9C%8B%E5%AF%AB%E8%A9%A6%E9%A1%8C.pdf)

#### 回答一：原始prompt

> 以下文字由ChatGPT 3.5生成

根据上文所述，森林需要缝隙是因为在激烈的生存竞争中，树木需要空间和光线来生长。密集的树木会阻碍阳光到达地面，导致新生命无法萌发，使得森林逐渐变得无生机。而缝隙提供了空间，让阳光穿透并滋养地面，为新生命的孕育创造了条件。

联想到人生，缝隙在我们的生活中也起着重要的作用。生活中的缝隙就像是转折点或者间隙，它们为我们提供了成长和改变的机会。正如森林需要缝隙来孕育新生命一样，人生也需要经历各种各样的缝隙来实现自我成长和转变。

在生活中，我们经常会遇到挫折和困难，这些看似消极的缝隙实际上是我们成长的机会。正如森林中倒下的树木会为新的生命留下空间一样，我们在挫折和失败中也能够发现新的可能性和方向。这些缝隙不仅是结束和开始的交汇处，也是由缺乏转化为希望的空间。

此外，缝隙也是我们思考和反思的时刻。在忙碌的生活中，我们往往会忽视内心的声音和需要。然而，当我们遇到缝隙时，我们有机会停下来，审视自己的生活，思考过去和未来，找到内心的平静和方向。这种自省和思考是我们成长和进步的关键。

综上所述，生活中的缝隙是我们成长、改变和思考的机会。正如森林需要缝隙来实现新生命的孕育一样，人生也需要经历各种各样的缝隙来实现个人成长和转变。因此，我们应该珍视缝隙，从中汲取力量，不断前行。

> 问题：没有结合生活经验，写出来比较空洞。


### 生成prompt示例

- Directly use the task description
> Write an essay with the topic {Topic}.

- Describe your needs
> Write a 3-paragraph essay about {Topic}. Please use a professional tone.

- Role-playing
> You are a perfect write. Please write an essay about {Topic}.

## Part 2: 用AI模型来评估文章

评估平台为[MediaTek Davinci平台](https://dvcbot.net/)，只提供给台大学生，旁听生看上去不可用。

提交的文章将根据讲课方提供的`evaluation prompts`来为文章打分。
