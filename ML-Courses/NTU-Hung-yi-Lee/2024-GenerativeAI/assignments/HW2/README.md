# GenAI HW2：都是AI的作文比赛

## 任务

- 用AI模型来生成两篇文章：中文/English（也可以自己撰写）
- 用AI模型来评估两篇文章：中文/English

## Part 1: 用AI模型来生成文章

### 生成英文文章

> Do you agree or disagree with the following statement?
> Artificial Intelligence will eventually replace humans in most areas of work in the future world.
> Write an English essay with around 300 words to express your instance with specific reasons and examples to support your answer.


### 生成中文文章

> 参考[113学年度学科能力测验试题：国语文写作能力测验 第二部分](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ceec.edu.tw/files/file_pool/1/0O021612121259980745/07-113%E5%AD%B8%E6%B8%AC%E5%9C%8B%E5%AF%AB%E8%A9%A6%E9%A1%8C.pdf)所附的文章，请以【缝隙的联想】为题，撰写一篇约五百字的繁体中文文章，并结合生活经验或见闻，书写你的感思与体悟。

- [113学年度学科能力测验试题：国语文写作能力测验 第二部分](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ceec.edu.tw/files/file_pool/1/0O021612121259980745/07-113%E5%AD%B8%E6%B8%AC%E5%9C%8B%E5%AF%AB%E8%A9%A6%E9%A1%8C.pdf)

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
