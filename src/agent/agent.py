import os
import json
import base64
import tempfile
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from PIL import Image
import requests
from dotenv import load_dotenv
import chainlit as cl

# ===================== 状态模型 =====================
class NodeHistory(BaseModel):
    """节点历史记录"""
    node_name: str = Field(description="节点名称")
    node_result: str = Field(description="节点返回结果")
    
class SingleImageState(BaseModel):
    """单张图片的遍历状态"""
    image_idx: int = Field(description="图片索引（0-based）")
    current_node: str = Field(default="root", description="当前检查节点")
    node_result: str = Field(default="", description="当前节点API返回结果")
    pending_nodes: List[str] = Field(default_factory=list, description="待检查节点队列")
    visited_nodes: Set[str] = Field(default_factory=set, description="已访问节点（防重入）")
    risks: List[str] = Field(default_factory=list, description="收集的风险项")
    rectifies: List[str] = Field(default_factory=list, description="收集的整改建议")
    is_finished: bool = Field(default=False, description="本图片检查是否完成")
    node_history: List[NodeHistory] = Field(default_factory=list, description="节点历史记录")

class MultiImageState(BaseModel):
    """多张图片的全局状态"""
    all_images_base64: List[str] = Field(description="所有图片的Base64")
    tree_config: Dict[str, Any] = Field(description="树形检查配置")
    
    # 每张图片的独立状态
    image_states: Dict[int, SingleImageState] = Field(default_factory=dict)
    
    # 全局进度控制
    completed_images: Set[int] = Field(default_factory=set, description="已完成的图片")
    total_images: int = Field(description="图片总数")
    
    # 最终聚合结果
    final_results: Dict[int, Dict[str, List[str]]] = Field(default_factory=lambda: {
        # 结构：{图片编号: {"risk": [], "rectify": []}}
    })

# ===================== 工具函数 =====================
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("❌ 请在.env文件中配置DASHSCOPE_API_KEY")

DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

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
    return f"data:image/{img_format};base64,{base64_str}"

def call_qwen_vl_api(image_base64: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    content = [
        {"type": "image", "image": image_base64},
        {"type": "text", "text": prompt}
    ]
    payload = {
        "model": "qwen3-vl-plus",
        "input": {"messages": [{"role": "user", "content": content}]},
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
        return pure_text.strip()
    except Exception as e:
        raise RuntimeError(f"❌ API调用失败：{str(e)}")

# ===================== 多图片API调用函数 =====================
def call_qwen_vl_with_multiple_images(image_base64_list: List[str], prompt: str) -> str:
    """调用Qwen-VL API处理多张图片"""
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建包含所有图片的content
    content = []
    
    # 添加所有图片
    for idx, img_base64 in enumerate(image_base64_list):
        content.append({"type": "image", "image": img_base64})
        # 添加图片编号提示（可选，帮助模型区分图片）
        # content.append({"type": "text", "text": f"图片{idx+1}："})
    
    # 添加文本提示
    content.append({"type": "text", "text": prompt})
    
    payload = {
        "model": "qwen3-vl-plus",
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {"temperature": 0.1, "result_format": "message"}
    }
    
    try:
        response = requests.post(DASHSCOPE_API_URL, headers=headers, json=payload, timeout=90)
        response_json = response.json()
        response.raise_for_status()
        choices = response_json["output"]["choices"]
        if not choices:
            raise RuntimeError("❌ 无有效回复")
        content_list = choices[0]["message"]["content"]
        pure_text = content_list[0]["text"] if (isinstance(content_list, list) and len(content_list) > 0) else str(content_list)
        return pure_text.strip()
    except Exception as e:
        raise RuntimeError(f"❌ 多图片API调用失败：{str(e)}")

# ===================== 智能上下文构建器 =====================
class ContextBuilder:
    """根据配置智能构建上下文"""
    
    @staticmethod
    def build_context(node_history: List[NodeHistory], context_type: str = "none", current_node: str = "") -> str:
        """
        构建上下文字符串
        
        Args:
            node_history: 节点历史记录
            context_type: 上下文类型 - none/parent/all
            current_node: 当前节点名称
        
        Returns:
            上下文字符串
        """
        if context_type == "none" or not node_history:
            return ""
        
        if context_type == "parent":
            # 只获取父节点信息
            if len(node_history) > 0:
                parent = node_history[-1]  # 最后一个就是父节点
                return f"📋 父节点检查结果：\n{parent.node_name}: {parent.node_result}\n\n"
            return ""
        
        elif context_type == "all":
            # 获取所有历史信息（除了当前节点）
            if not node_history:
                return ""
            
            context_lines = ["📋 检查历史记录："]
            for i, history in enumerate(node_history):
                if history.node_result:
                    # 美化显示
                    display_name = history.node_name
                    display_result = history.node_result
                    
                    # 特殊处理根节点
                    if history.node_name == "root":
                        display_name = "初始检查"
                        if display_result and display_result != "无":
                            display_result = f"发现：{display_result}"
                    
                    context_lines.append(f"{i+1}. {display_name}: {display_result}")
            
            if len(context_lines) > 1:
                return "\n".join(context_lines) + "\n\n"
        
        return ""
    
    @staticmethod
    def build_enhanced_prompt(
        node_history: List[NodeHistory],
        base_prompt: str,
        node_name: str,
        context_type: str = "none"
    ) -> str:
        """
        构建增强的prompt
        
        Args:
            node_history: 节点历史
            base_prompt: 基础prompt
            node_name: 当前节点名称
            context_type: 上下文类型
        
        Returns:
            增强的prompt
        """
        # 构建上下文
        context = ContextBuilder.build_context(node_history, context_type, node_name)
        
        if not context:
            return base_prompt
        
        # 根据节点类型添加指导语
        guidance = ""
        
        # 检查节点名称，添加特定指导
        if "cable" in node_name.lower():
            guidance = "\n注意：请基于之前的电缆检查结果进行判断。"
        elif "box" in node_name.lower() or "配电" in node_name.lower():
            guidance = "\n注意：请基于配电箱的检查情况进行综合评估。"
        
        # 构建最终prompt
        enhanced_prompt = f"""{context}根据以上检查历史，现在需要进行下一步检查。

{base_prompt}{guidance}

请结合历史检查结果，给出准确的判断。"""
        
        return enhanced_prompt

# ===================== 核心节点 =====================
def process_image_node(state: MultiImageState) -> Dict:
    """处理单张图片的当前节点"""
    # 创建新的状态对象
    new_state = MultiImageState(
        all_images_base64=state.all_images_base64.copy(),
        tree_config=state.tree_config.copy(),
        image_states={k: SingleImageState(**v.dict()) for k, v in state.image_states.items()},
        completed_images=state.completed_images.copy(),
        total_images=state.total_images,
        final_results=state.final_results.copy()
    )
    
    # 找出需要处理的图片
    active_images = [
        idx for idx, img_state in new_state.image_states.items()
        if not img_state.is_finished and idx not in new_state.completed_images
    ]
    
    if not active_images:
        aggregate_results(new_state)
        return new_state.dict()
    
    # 处理第一张活跃图片
    current_idx = active_images[0]
    img_state = new_state.image_states[current_idx]
    
    print(f"🔄 处理图片{current_idx+1} - 当前节点：{img_state.current_node}")
    
    # 检查是否已访问过该节点
    if img_state.current_node in img_state.visited_nodes:
        print(f"⚠️  图片{current_idx+1}节点{img_state.current_node}已访问，跳过")
        if img_state.pending_nodes:
            img_state.current_node = img_state.pending_nodes.pop(0)
        else:
            img_state.is_finished = True
            new_state.completed_images.add(current_idx)
        return new_state.dict()
    
    img_state.visited_nodes.add(img_state.current_node)
    
    # 获取节点配置
    image_base64 = new_state.all_images_base64[current_idx]
    node_config = new_state.tree_config.get(img_state.current_node, {})
    
    # 记录当前节点到历史（在执行前记录）
    current_history = NodeHistory(
        node_name=img_state.current_node,
        node_result=""  # 初始为空
    )
    img_state.node_history.append(current_history)
    
    # 获取上下文类型配置（默认为"none"）
    context_type = node_config.get("context", "none")
    print(f"📝 节点{img_state.current_node}的上下文类型：{context_type}")
    
    # 处理根节点
    if img_state.current_node == "root":
        prompt = node_config.get("prompt", "")
        if prompt:
            result = call_qwen_vl_api(image_base64, prompt)
            img_state.node_result = result
            current_history.node_result = result
            
            print(f"📌 图片{current_idx+1} root节点返回：{result}")
            
            # 解析返回的元素
            elements = [
                elem.strip() for elem in result.split(",") 
                if elem.strip() and elem.strip() != "无"
            ]
            
            # 映射到子节点
            child_map = node_config.get("child_map", {})
            for element in elements:
                next_node = child_map.get(element)
                if next_node and next_node not in img_state.visited_nodes:
                    img_state.pending_nodes.append(next_node)
            
            print(f"📌 图片{current_idx+1} 生成待处理节点：{img_state.pending_nodes}")
            
        # 移动到下一个节点
        if img_state.pending_nodes:
            img_state.current_node = img_state.pending_nodes.pop(0)
        else:
            img_state.is_finished = True
            new_state.completed_images.add(current_idx)
        
        return new_state.dict()
    
    # 处理非根节点
    base_prompt = node_config.get("prompt", "")
    
    if base_prompt:
        # 构建增强prompt（排除当前节点）
        history_for_context = img_state.node_history[:-1]
        
        enhanced_prompt = ContextBuilder.build_enhanced_prompt(
            node_history=history_for_context,
            base_prompt=base_prompt,
            node_name=img_state.current_node,
            context_type=context_type
        )
        
        # 打印上下文信息（调试用）
        if context_type != "none" and history_for_context:
            print(f"📋 图片{current_idx+1} 使用的上下文：")
            for hist in history_for_context[-3:]:  # 只显示最近3个
                print(f"   - {hist.node_name}: {hist.node_result[:50]}...")
        
        result = call_qwen_vl_api(image_base64, enhanced_prompt)
        img_state.node_result = result
        current_history.node_result = result
        
        print(f"📌 图片{current_idx+1} 节点{img_state.current_node}返回：{result}")
    else:
        # 如果没有prompt，直接使用空结果
        img_state.node_result = ""
        current_history.node_result = "无prompt节点"
        print(f"📌 图片{current_idx+1} 节点{img_state.current_node}（无prompt）")
    
    # 收集风险和建议
    risk = node_config.get("risk", "").strip()
    rectify = node_config.get("rectify", "").strip()
    
    if risk and risk != "无" and risk not in img_state.risks:
        img_state.risks.append(risk)
        print(f"✅ 图片{current_idx+1} 收集风险：{risk[:50]}...")
    
    if rectify and rectify != "无" and rectify not in img_state.rectifies:
        img_state.rectifies.append(rectify)
        print(f"✅ 图片{current_idx+1} 收集整改建议：{rectify[:50]}...")
    
    # 处理子节点映射
    child_map = node_config.get("child_map", {})
    next_node = child_map.get(img_state.node_result.strip())
    
    if next_node and next_node not in img_state.visited_nodes:
        img_state.current_node = next_node
        print(f"📌 图片{current_idx+1} 映射到子节点：{next_node}")
    elif img_state.pending_nodes:
        img_state.current_node = img_state.pending_nodes.pop(0)
        print(f"📌 图片{current_idx+1} 从队列取下一个节点：{img_state.current_node}")
    else:
        img_state.is_finished = True
        new_state.completed_images.add(current_idx)
        print(f"📌 图片{current_idx+1} 检查完成")
    
    # 检查是否所有图片都完成了
    if len(new_state.completed_images) >= new_state.total_images:
        print("🔚 所有图片处理完成，开始聚合结果...")
        aggregate_results(new_state)
    
    return new_state.dict()

def aggregate_results(state: MultiImageState):
    """聚合所有图片的结果到final_results"""
    for idx, img_state in state.image_states.items():
        pic_num = idx + 1
        
        if pic_num not in state.final_results:
            state.final_results[pic_num] = {"risk": [], "rectify": []}
        
        # 去重添加风险和整改建议
        for risk in img_state.risks:
            if risk and risk not in state.final_results[pic_num]["risk"]:
                state.final_results[pic_num]["risk"].append(risk)
        
        for rectify in img_state.rectifies:
            if rectify and rectify not in state.final_results[pic_num]["rectify"]:
                state.final_results[pic_num]["rectify"].append(rectify)

# ===================== 路由函数 =====================
def multi_image_router(state: MultiImageState) -> Literal["process_node", "__end__"]:
    if len(state.completed_images) >= state.total_images:
        return "__end__"
    
    active_images = [
        idx for idx, img_state in state.image_states.items()
        if not img_state.is_finished and idx not in state.completed_images
    ]
    
    if active_images:
        current_idx = active_images[0]
        img_state = state.image_states[current_idx]
        
        if img_state.current_node and img_state.current_node in state.tree_config:
            return "process_node"
        
        if img_state.pending_nodes:
            img_state.current_node = img_state.pending_nodes.pop(0)
            return "process_node"
        
        img_state.is_finished = True
        state.completed_images.add(current_idx)
        return "process_node"
    
    return "__end__"

# ===================== 构建Graph =====================
def build_multi_image_graph(tree_config: Dict[str, Any]) -> CompiledStateGraph:
    graph = StateGraph(MultiImageState)
    graph.add_node("process_node", process_image_node)
    graph.add_edge(START, "process_node")
    graph.add_conditional_edges(
        "process_node",
        multi_image_router,
        {"process_node": "process_node", "__end__": END}
    )
    return graph.compile()

# ===================== 格式化结果 =====================
def format_check_result(check_results: Dict[int, Dict[str, List[str]]]) -> str:
    if not check_results:
        return "✅ 未检测到任何图片的检查结果"
    
    final_output = "📋 施工现场用电安全检查结果（按图片编号）\n"
    final_output += "="*80 + "\n"
    
    for pic_num in sorted(check_results.keys()):
        res = check_results[pic_num]
        risks = res.get("risk", [])
        rectifies = res.get("rectify", [])
        
        final_output += f"🖼️  第{pic_num}张图片\n"
        
        if not risks:
            final_output += "   ✅ 未发现安全风险，符合施工用电安全规范要求\n"
        else:
            final_output += "   ⚠️  发现安全风险：\n"
            max_len = max(len(risks), len(rectifies))
            for idx in range(max_len):
                risk = risks[idx] if idx < len(risks) else "未知风险"
                rectify = rectifies[idx] if idx < len(rectifies) else "请根据现场情况制定整改措施"
                final_output += f"      {idx+1}. 风险隐患：{risk}\n"
                final_output += f"         整改建议：{rectify}\n"
        
        final_output += "-"*80 + "\n"
    
    return final_output.strip()

# ===================== Agent核心 =====================
class MultiImageTreeAgent:
    def __init__(self, config_path: str = "tree_check_config.json"):
        self.tree_config = self.load_config(config_path)
        self.graph = build_multi_image_graph(self.tree_config)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 为所有节点添加默认的context字段
        for node_name, node_config in config.items():
            if "context" not in node_config:
                config[node_name]["context"] = "none"
        
        return config
    
    def run(self, image_paths: List[str], question: Optional[str] = None) -> str:
        if not image_paths:
            raise ValueError("❌ 请至少提供1张图片")
        if len(image_paths) > 5:
            raise ValueError("❌ 最多支持5张图片")
        
        # 处理自定义问题
        if question and question.strip():
            # 将所有图片转换为Base64
            image_base64_list = [image_to_base64_with_prefix(path) for path in image_paths]
            
            # 构建明确的prompt
            prompt = f"""请根据上传的{len(image_base64_list)}张图片，综合分析并回答问题。

问题：{question}

要求：
1. 请仔细分析每一张图片的内容
2. 结合所有图片进行综合判断
3. 如果图片间存在差异或联系，请明确指出
4. 给出基于所有图片的完整答案

现在开始分析："""
            
            # 调用多图片API
            answer = call_qwen_vl_with_multiple_images(image_base64_list, prompt)
            return answer
        
        # 树形检查
        image_base64_list = [image_to_base64_with_prefix(path) for path in image_paths]
        
        # 初始化每张图片的状态
        image_states = {}
        for idx in range(len(image_paths)):
            image_states[idx] = SingleImageState(
                image_idx=idx,
                current_node="root",
                pending_nodes=[],
                visited_nodes=set(),
                risks=[],
                rectifies=[],
                is_finished=False
            )
        
        # 构建初始状态
        initial_state = MultiImageState(
            all_images_base64=image_base64_list,
            tree_config=self.tree_config,
            image_states=image_states,
            completed_images=set(),
            total_images=len(image_paths),
            final_results={}
        )
        
        print(f"🚀 开始执行树形检查，共{len(image_paths)}张图片")
        
        try:
            final_state_dict = self.graph.invoke(initial_state)
            final_state = MultiImageState(**final_state_dict)
            
            print(f"✅ 检查完成，共处理{len(final_state.completed_images)}张图片")
            
            return format_check_result(final_state.final_results)
            
        except Exception as e:
            print(f"❌ 执行Graph失败：{e}")
            import traceback
            traceback.print_exc()
            return f"❌ 检查过程出错：{str(e)}"

# ===================== Chainlit集成 =====================
agent = None

@cl.on_chat_start
async def start_chat():
    global agent
    try:
        agent = MultiImageTreeAgent()
        await cl.Message(
            content="""🎉 欢迎使用施工现场用电安全检查Agent！
✅ 使用方式1（自动树形检查）：上传1~5张图片 → 直接发送（无需输入文字）
✅ 使用方式2（自定义提问）：上传图片后输入问题 → 发送
✅ 支持格式：JPG/PNG，最多5张图片
✅ 多图片提问：上传多张图片并提问，系统会综合分析所有图片

🔍 支持智能上下文传递：通过配置文件控制节点是否需要上下文"""
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Agent初始化失败：{str(e)}").send()

@cl.on_message
async def handle_message(message: cl.Message):
    global agent
    try:
        if agent is None:
            agent = MultiImageTreeAgent()
        
        image_paths = []
        
        if hasattr(message, 'elements') and message.elements:
            for element in message.elements:
                is_image = False
                
                if hasattr(element, 'type') and element.type == 'image':
                    is_image = True
                elif hasattr(element, 'mime') and element.mime:
                    if isinstance(element.mime, str) and element.mime.startswith('image/'):
                        is_image = True
                elif hasattr(element, 'name') and element.name:
                    name = element.name.lower()
                    if any(name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        is_image = True
                
                if is_image and hasattr(element, 'path') and element.path and os.path.exists(element.path):
                    image_paths.append(element.path)
        
        if not image_paths:
            await cl.Message(content="❌ 未检测到有效图片！请上传图片。").send()
            return
        
        if len(image_paths) > 5:
            await cl.Message(content="❌ 最多支持5张图片").send()
            return
        
        user_question = message.content.strip() if message.content else ""
        
        if user_question:
            # 多图片自定义问题
            if len(image_paths) == 1:
                await cl.Message(content="🔍 正在根据问题分析图片...").send()
            else:
                await cl.Message(content=f"🔍 正在综合分析{len(image_paths)}张图片，请稍候...").send()
        else:
            # 树形检查
            await cl.Message(content=f"🔍 正在智能检查{len(image_paths)}张图片，请稍候...").send()
        
        result = agent.run(image_paths, user_question)
        
        await cl.Message(content=result).send()
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ 完整错误信息:\n{error_detail}")
        await cl.Message(content=f"❌ 处理失败：{str(e)}").send()

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("✅ 施工现场用电安全检查Agent初始化完成！")
    print("✅ LangGraph版本: 1.0.7")
    print("✅ Chainlit版本: 2.9.6")
    print("✅ 支持配置驱动的智能上下文")
    print("✅ 支持多图片综合分析（自定义问题）")
    print("✅ 执行命令启动: chainlit run 本文件.py")
    print("=" * 60)