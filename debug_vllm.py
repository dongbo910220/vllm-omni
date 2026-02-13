#!/usr/bin/env python3
"""
Debug 启动脚本 - 避免 uvloop 兼容性问题
"""
import os
import sys
import asyncio

# 强制禁用 uvloop - 必须在任何导入之前设置
os.environ['VLLM_USE_UVLOOP'] = '0'
# 允许网络下载模型
print(f"🔧 设置 VLLM_USE_UVLOOP=0: {os.environ.get('VLLM_USE_UVLOOP')}")
print(f"🌐 允许网络下载模型")

# 设置参数 - 使用本地1.5B模型，针对内存优化
sys.argv = [
    'debug_vllm.py',
    '--model', '/home/bdong/Ai_project/models/Qwen2-1.5B-Instruct',  # 使用本地1.5B模型
    '--port', '8000',
    '--max-model-len', '128',  # 大幅减小最大长度
    '--gpu-memory-utilization', '0.85',  # 提高GPU内存利用率
    '--dtype', 'half',  # 使用半精度节省内存
    '--enforce-eager',  # ✅ 禁用 CUDA Graph，允许在 forward 中打断点
]

print(f"🚀 启动参数: {sys.argv}")

if __name__ == '__main__':
    # 复制 api_server.py 的启动逻辑，但使用 asyncio.run 而不是 uvloop.run
    from vllm.entrypoints.openai.api_server import (
        cli_env_setup, FlexibleArgumentParser, make_arg_parser,
        validate_parsed_serve_args, run_server
    )

    print("🔧 启动 vLLM API 服务器 (使用 asyncio 而不是 uvloop)")

    # 初始化环境
    cli_env_setup()

    # 解析参数
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # 使用标准 asyncio.run 而不是 uvloop.run
    asyncio.run(run_server(args))