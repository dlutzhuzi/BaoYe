import os
import json
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
from langgraph.graph import StateGraph, MessagesState, START, END
import chainlit as cl

# ===================== 1. 加载外部检查规则文件 =====================
def load_safety_rules(file_path: str = "manual_checklist.json") -> list:
    """加载外部的施工用电安全检查规则JSON文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 检查规则文件不存在：{file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        if not isinstance(rules, list) or len(rules) == 0:
            raise ValueError("❌ 检查规则文件格式错误，必须是非空列表")
        return rules
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ 检查规则文件JSON解析失败：{str(e)}")
    except Exception as e:
        raise RuntimeError(f"❌ 加载检查规则失败：{str(e)}")

# 加载规则（全局变量）
SAFETY_RULES = load_safety_rules()

# ===================== 2. 基础配置与工具函数 =====================
# 加载环境变量
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 请配置DASHSCOPE_API_KEY")

DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 图片转带前缀的Base64（兼容本地路径/Chainlit图片）
def image_to_base64_with_prefix(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 图片不存在：{image_path}")
    
    try:
        with Image.open(image_path) as img:
            img_format = img.format.lower()
            if img_format == "jpg":
                img_format = "jpeg"
            if img_format not in ["jpeg", "png"]:
                raise ValueError(f"❌ 不支持的图片格式：{img.format}，仅支持JPG/PNG")
    except Exception as e:
        raise ValueError(f"❌ 识别图片格式失败：{str(e)}")
    
    with open(image_path, "rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8").strip()
    base64_with_prefix = f"data:image/{img_format};base64,{base64_str}"
    return base64_with_prefix

# ===================== 3. 核心：调用阿里云多模态API =====================
def call_qwen_vl_api(image_base64_list: list, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    content = []
    for img_b64 in image_base64_list:
        content.append({"type": "image", "image": img_b64})
    content.append({"type": "text", "text": prompt})
    
    payload = {
        "model": "qwen3-vl-plus",
        "input": {
            "messages": [{"role": "user", "content": content}]
        },
        "parameters": {"temperature": 0.1, "result_format": "message"}
    }

    try:
        response = requests.post(DASHSCOPE_API_URL, headers=headers, json=payload, timeout=60)
        response_json = response.json()
        response.raise_for_status()
        
        choices = response_json["output"]["choices"]
        if not choices:
            raise RuntimeError("❌ 无有效回复")
        content_list = choices[0]["message"]["content"]
        pure_text = content_list[0]["text"] if (isinstance(content_list, list) and len(content_list) > 0) else str(content_list)
        
        return pure_text
    
    except requests.exceptions.HTTPError as e:
        err_code = response_json.get("code", "未知")
        err_msg = response_json.get("message", "未知")
        raise RuntimeError(f"❌ API失败（状态码：{response.status_code}）\n错误码：{err_code}\n错误信息：{err_msg}")
    except Exception as e:
        raise RuntimeError(f"❌ 调用异常：{str(e)}")

# ===================== 4. LangGraph节点函数（兼容两种模式） =====================
def multimodal_agent_node(state: MessagesState):
    try:
        user_msg = state["messages"][0]
        image_base64_list = []
        user_question = None
        
        # 提取图片和用户文字（如果有）
        for item in user_msg.content:
            if item["type"] == "image_base64":
                image_base64_list.append(item["image_base64"])
            elif item["type"] == "text":
                user_question = item["text"].strip()
        
        # 校验图片数量
        if not image_base64_list:
            raise ValueError("❌ 未检测到图片")
        if len(image_base64_list) > 5:
            raise ValueError("❌ 最多支持5张图片")
        
        # 构造提示词：有用户问题则用用户问题，无则自动检查
        if user_question and user_question != "":
            # 模式1：用户自定义提问
            prompt = f"请根据上传的图片，回答以下问题：{user_question}\n要求：回答准确、简洁，基于图片内容客观回复。"
        else:
            # 模式2：自动按规则检查
            rules_text = "\n".join([
                f"{idx+1}. 【{item['大类']}】{item['检查子项']}\n"
                f"   判断标准：{item['判断标准']}\n"
                f"   合规要求：{item['合规要求']}"
                for idx, item in enumerate(SAFETY_RULES)
            ])
            
            prompt = f"""请你作为施工用电安全规范检查专家，根据以下规则逐项检查上传的图片内容：

{rules_text}

检查要求：
1. 严格按照每个检查子项的判断标准，判断图片中对应的内容是否合规；
2. 对于每个检查子项，明确输出「合规」「不合规」或「未涉及」；
3. 如果判断为「不合规」，请简要说明违反的具体问题；
4. 你要针对每张图片分别给出判断，不要把答案混在一起；
5. 最终输出格式为表格形式，包含以下列：图片编号、检查结果、不合规说明（如适用）；
"""
        
        # 调用API生成回复
        ai_answer = call_qwen_vl_api(image_base64_list, prompt)
        return {"messages": [{"role": "ai", "content": ai_answer}]}
    
    except Exception as e:
        print(f"❌ 节点失败：{str(e)}")
        raise

# ===================== 5. 构建LangGraph工作流 =====================
graph = StateGraph(MessagesState)
graph.add_node("multimodal_agent", multimodal_agent_node)
graph.add_edge(START, "multimodal_agent")
graph.add_edge("multimodal_agent", END)
compiled_graph = graph.compile()

# ===================== 6. Agent调用函数（兼容有无问题） =====================
def run_agent(image_paths: list, question: str = None):
    if not image_paths or len(image_paths) == 0:
        raise ValueError("❌ 请至少提供1张图片")
    if len(image_paths) > 5:
        raise ValueError("❌ 最多支持5张图片")

    image_base64_list = [image_to_base64_with_prefix(img_path) for img_path in image_paths]

    # 构造用户消息内容（图片+文字/仅图片）
    contents = []
    for b64 in image_base64_list:
        contents.append({"type": "image_base64", "image_base64": b64})
    if question and question.strip() != "":
        contents.append({"type": "text", "text": question.strip()})

    user_message = {"role": "user", "content": contents}
    result = compiled_graph.invoke({"messages": [user_message]})

    ai_msg = result["messages"][-1]
    final_answer = ai_msg.content if hasattr(ai_msg, "content") else (ai_msg["content"] if isinstance(ai_msg, dict) else str(ai_msg))

    return final_answer

# ===================== 7. Chainlit交互逻辑（灵活交互） =====================
@cl.on_chat_start
async def start_chat():
    """初始化：提示两种使用方式"""
    await cl.Message(
        content="""🎉 欢迎使用施工现场用电安全检查Agent！
✅ 使用方式1（自动检查）：点击输入框左侧「📎」→ 选择「Images」→ 上传1~5张图片 → 直接发送（无需输入文字）
✅ 使用方式2（自定义提问）：上传图片后，在输入框中输入具体问题（如“检查配电箱是否上锁”）→ 发送
✅ 支持格式：JPG/PNG，最多5张图片"""
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """处理消息：兼容「仅图片」和「图片+文字提问」两种场景"""
    try:
        # 1. 提取图片元素
        image_elements = []
        for elem in message.elements:
            if isinstance(elem, cl.Image) or (isinstance(elem, cl.File) and getattr(elem, 'mime_type', '').startswith('image/')):
                image_elements.append(elem)
        
        # 2. 校验图片
        if not image_elements:
            await cl.Message(
                content="""❌ 未检测到有效图片！请按以下步骤操作：
1. 点击输入框左侧的「📎」图标（附件图标）；
2. 选择「Images」选项（而非「Files」）；
3. 上传1~5张JPG/PNG格式的施工现场图片；
4. 可直接发送（自动检查）或输入问题后发送（自定义提问）。"""
            ).send()
            return
        
        if len(image_elements) > 5:
            await cl.Message(content=f"❌ 上传了{len(image_elements)}张图片，最多仅支持5张！请重新上传。").send()
            return
        
        # 3. 提取图片路径
        image_paths = []
        for img_elem in image_elements:
            if hasattr(img_elem, 'path') and os.path.exists(img_elem.path):
                image_paths.append(img_elem.path)
            else:
                await cl.Message(content=f"❌ 图片{getattr(img_elem, 'name', '未知')}路径无效！").send()
                return
        
        # 4. 提取用户文字提问（可能为空）
        user_question = message.content.strip() if hasattr(message, 'content') and message.content else ""
        
        # 5. 调用Agent（根据有无提问自动适配模式）
        await cl.Message(content=f"🔍 正在分析{len(image_elements)}张图片，请稍候...").send()
        result = run_agent(image_paths, user_question)
        
        # 6. 发送结果
        await cl.Message(content=result).send()
    
    except Exception as e:
        error_info = f"❌ 处理失败：{str(e)}"
        await cl.Message(content=error_info).send()