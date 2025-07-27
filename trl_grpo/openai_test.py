from openai import OpenAI

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-LjjwKIT8YaKGAY4uk2aYmXT98x1n2JI9fMhyB3EqAIZcMd7F',
)

text1 = "你好，我是小明，很高兴认识你。"
text2 = "你好，我是小红，很高兴认识你。"
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "你是一个专业的文本相似度评估专家。请分析两个文本的关键语义相似度，并给出0到1之间的分数，其中0表示完全不相似，1表示完全相同。请只返回数字，不要其他解释。"
        },
        {
            "role": "user",
            "content": f"请评估以下两个文本的语义相似度，给出0-1之间的分数：\n\n文本1：{text1}\n\n文本2：{text2}"
        }
    ],
    temperature=0.1,
    max_completion_tokens=10

)

# 提取分数
score_text = response.choices[0].message.content

try:
    score = float(score_text)
    # 确保分数在0-1范围内
    score = max(0.0, min(1.0, score))
    print(score)
except ValueError:
    print(f"Warning: Could not parse similarity score '{score_text}'. Using fallback score.")
    print(0.5)