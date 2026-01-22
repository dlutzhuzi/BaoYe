import os
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
from langgraph.graph import StateGraph, MessagesState, START, END
import chainlit as cl


# 1. 加载环境变量
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 请配置DASHSCOPE_API_KEY")

DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 2. 辅助函数：图片转带前缀的Base64（兼容本地路径/Chainlit图片）
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

# 3. 核心：调用阿里云多模态API
def call_qwen_vl_api(image_base64_list: list, question: str) -> str:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    content = []
    for img_b64 in image_base64_list:
        content.append({"type": "image", "image": img_b64})
    content.append({"type": "text", "text": question})
    
    payload = {
        "model": "qwen-vl-plus",
        "input": {
            "messages": [{"role": "user", "content": content}]
        },
        "parameters": {"temperature": 0.5, "result_format": "message"}
    }

    try:
        response = requests.post(DASHSCOPE_API_URL, headers=headers, json=payload, timeout=30)
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

# 4. LangGraph节点函数
def multimodal_agent_node(state: MessagesState):
    try:
        user_msg = state["messages"][0]
        image_base64_list = []
        question = None
        
        for item in user_msg.content:
            if item["type"] == "image_base64":
                image_base64_list.append(item["image_base64"])
            elif item["type"] == "text":
                question = item["text"]
        
        if not image_base64_list:
            raise ValueError("❌ 未检测到图片")
        if not question:
            raise ValueError("❌ 未检测到问题")
        if len(image_base64_list) > 5:
            raise ValueError("❌ 最多支持5张图片")
        
        ai_answer = call_qwen_vl_api(image_base64_list, question)
        return {"messages": [{"role": "ai", "content": ai_answer}]}
    
    except Exception as e:
        print(f"❌ 节点失败：{str(e)}")
        raise

# 5. 构建LangGraph工作流
graph = StateGraph(MessagesState)
graph.add_node("multimodal_agent", multimodal_agent_node)
graph.add_edge(START, "multimodal_agent")
graph.add_edge("multimodal_agent", END)
compiled_graph = graph.compile()

# 6. Agent调用函数
def run_agent(image_paths: list, question: str):
    if not image_paths or len(image_paths) == 0:
        raise ValueError("❌ 请至少提供1张图片")
    if len(image_paths) > 5:
        raise ValueError("❌ 最多支持5张图片")

    image_base64_list = [image_to_base64_with_prefix(img_path) for img_path in image_paths]

    contents = []
    for b64 in image_base64_list:
        contents.append({"type": "image_base64", "image_base64": b64})
    contents.append({"type": "text", "text": question})

    user_message = {"role": "user", "content": contents}
    result = compiled_graph.invoke({"messages": [user_message]})

    ai_msg = result["messages"][-1]
    final_answer = ai_msg.content if hasattr(ai_msg, "content") else (ai_msg["content"] if isinstance(ai_msg, dict) else str(ai_msg))

    return final_answer

# 7. Chainlit核心交互逻辑
@cl.on_chat_start
async def start_chat():
    """初始化：明确提示上传方式"""
    await cl.Message(
        content="""🎉 欢迎使用施工现场检查Agent！
✅ 上传图片方式：点击输入框左侧「📎」图标 → 选择「Images」→ 上传1~5张JPG/PNG图片
✅ 输入问题后发送，即可分析所有图片的供电箱/杂物堆放情况"""
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """处理消息：适配Chainlit v1.x 图片元素"""
    try:
        # ========== 核心修正：Chainlit v1.x 图片元素识别 ==========
        # 步骤1：打印调试信息（方便排查）
        #await cl.Message(content=f"🔍 调试：收到的元素总数={len(message.elements)}，元素详情={[{'type': type(e), 'name': getattr(e, 'name', '无'), 'mime': getattr(e, 'mime_type', '无')} for e in message.elements]}").send()
        
        # 步骤2：筛选图片元素（v1.x 优先识别 cl.Image 类型）
        image_elements = []
        # 兼容两种情况：cl.Image 类型 / File类型（兜底）
        for elem in message.elements:
            # 情况1：Chainlit v1.x 上传图片的原生类型（核心）
            if isinstance(elem, cl.Image):
                image_elements.append(elem)
            # 情况2：兜底兼容 File 类型
            elif isinstance(elem, cl.File) and getattr(elem, 'mime_type', '').startswith('image/'):
                image_elements.append(elem)
        
        # 步骤3：校验图片数量
        if not image_elements:
            await cl.Message(
                content="""❌ 未检测到有效图片！请按以下步骤操作：
1. 点击输入框左侧的「📎」图标（附件图标）；
2. 选择「Images」选项（而非「Files」）；
3. 上传1~5张JPG/PNG格式的施工现场图片；
4. 输入问题后重新发送。"""
            ).send()
            return
        
        if len(image_elements) > 5:
            await cl.Message(content=f"❌ 上传了{len(image_elements)}张图片，最多仅支持5张！请重新上传。").send()
            return
        
        # 步骤4：提取所有图片的本地路径（v1.x 图片元素的 path 属性）
        image_paths = []
        for img_elem in image_elements:
            # cl.Image 和 cl.File 都有 path 属性
            if hasattr(img_elem, 'path') and os.path.exists(img_elem.path):
                image_paths.append(img_elem.path)
            else:
                await cl.Message(content=f"❌ 图片{getattr(img_elem, 'name', '未知')}路径无效！").send()
                return
        
        # 步骤5：调用Agent并返回结果
        answer = run_agent(image_paths, message.content)
        await cl.Message(content=f"✅ 已分析{len(image_elements)}张图片，检查结果：\n\n{answer}").send()
    
    except Exception as e:
        await cl.Message(content=f"❌ 处理失败：{str(e)}\n🔍 错误详情：{e.__traceback__}").send()