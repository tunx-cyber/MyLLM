'''
最简易的分词器是把文章所有的字符或者单词当作一个token，然后给予索引。
本质就是给每个token一个唯一的ID，形成字典的结构。
'''

'''
实际上肯定不会是上面这么简单的分词，而是基于字节对编码（Byte Pair Encoding, BPE）算法来进行分词。
BPE算法的核心思想是将文本中的字符对进行频繁替换，直到达到预设的词汇表大小。具体步骤如下：
1. 初始化词汇表：将文本中的所有字符作为初始词汇表。
2. 统计字符对频率：计算文本中所有相邻字符对的频率。
3. 替换频率最高的字符对：将频率最高的字符对替换为一个新的字符，更新词汇表。
4. 重复步骤2和3，直到达到预设的词汇表大小。
在实际应用中，BPE算法通常会结合其他技术，如子词分割、词干提取等，以提高分词的准确性和效率。
BPE算法的优点：
1. 可以处理未登录词：BPE算法可以将未登录词分解为已登录词的组合，从而提高模型的泛化能力。
2. 适应性强：BPE算法可以根据数据集的特点自动调整词汇表大小，从而提高模型的性能。
BPE算法的缺点：
1. 计算复杂度高：BPE算法需要频繁地计算字符对的频率，计算复杂度较高。
2. 训练时间长：BPE算法需要对大量文本进行训练，训练时间较长。
3. 词汇表大小不确定：BPE算法的词汇表大小是一个超参数，需要根据数据集的特点进行调整。
'''
from importlib.metadata import version
import tiktoken
# print("tiktoken版本:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
# text = (
#  "Hello, do you like tea?  " 
#  "In the sunlit terraces"
#  "of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={""})
# print(integers)
# print(tokenizer.decode(integers))