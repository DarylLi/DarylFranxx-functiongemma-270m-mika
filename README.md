# functiongemma-270m-mika

基于 [lmstudio-community/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) 修改发布。

🏠 **仓库地址 → [DarylFranxx/functiongemma-270m-mika](https://huggingface.co/DarylFranxx/functiongemma-270m-mika)**

🎮 **在线体验 Function Calling Demo → [DarylFranxx/functiongemma-270m-demo](https://darylfranxx-functiongemma-270m-mika-demo.hf.space/?logs=container&__theme=system)**

---

## 架构信息

| 参数        | 值                                 |
| ----------- | ---------------------------------- |
| 架构        | Gemma3ForCausalLM                  |
| 参数量      | ~270M                              |
| Hidden Size | 640                                |
| 层数        | 18 (sliding + full attention 混合) |
| 词表大小    | 262,144                            |
| 最大上下文  | 32,768 tokens                      |
| 精度        | bfloat16                           |

## 修改内容

- ✅ 优化了生成参数配置（temperature / top_p / top_k）
- ✅ 新增 `generation_config.json`
- ✅ 优化了支持 Function Calling 的 Chat Template
- ✅ 提升 max_length 至 8192

---

## 快速开始

### Function Calling 示例（推荐用法）

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json

model_id = "DarylFranxx/functiongemma-270m-mika"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

system_prompt = f"You are a tool caller.\nTools: {json.dumps(tools, ensure_ascii=False)}"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "上海天气如何？"}
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,
        temperature=0.01,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
# 预期输出: {"name": "get_weather", "parameters": {"location": "上海"}}
```

### MLX 方式（Mac Apple Silicon）

```python
from mlx_lm import load, generate
import json

model, tokenizer = load("DarylFranxx/functiongemma-270m-mika")

tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名称"}
            },
            "required": ["location"]
        }
    }
]

system_prompt = f"You are a tool caller.\nTools: {json.dumps(tools, ensure_ascii=False)}"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": "上海天气如何？"}
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, max_tokens=256)
print(response)
```

---

## 注意事项

- 需要 `transformers >= 4.57.3` 才支持 `Gemma3ForCausalLM`
- MLX 格式仅适用于 Apple Silicon（M1 / M2 / M3 / M4）
- Function Calling 建议 `temperature=0` 以获得稳定的 JSON 输出
- 遵守原始模型 Apache 2.0 许可证
