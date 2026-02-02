## Why

具身智能体（机器人、无人机等）需要实时理解环境并规划导航路径。现有方案通常依赖传统 SLAM 或预定义规则，缺乏对复杂场景的语义理解能力。通过集成大语言模型（如 Gemini）的视觉理解能力，可以实现更智能的环境感知和任务导向的路径规划。

## What Changes

- 新增视频流/摄像头输入模块，支持实时抽帧
- 新增 LLM API 客户端，支持 Gemini 原生格式和 OpenAI 兼容格式
- 新增环境分析模块，输出场景中的关键物体和障碍物
- 新增任务分析模块，根据环境和目标生成行动意图
- 新增路径点生成模块，输出 5 个导航点坐标（从底部中间向上等间隔分布）
- 新增可视化模块，展示分析结果和路径点（近绿远红渐变）

## Capabilities

### New Capabilities

- `video-capture`: 视频流/摄像头输入，抽帧存入长度为 1 的队列
- `llm-client`: LLM API 客户端，支持 Gemini 和 OpenAI 兼容格式
- `scene-analysis`: 环境分析，识别场景中的物体、障碍物、可通行区域
- `task-reasoning`: 任务推理，根据环境和目标生成行动意图
- `waypoint-generation`: 路径点生成，输出 5 个导航点的 x/y 坐标
- `visualization`: 可视化展示，框体标注 + 路径曲线 + 颜色渐变

### Modified Capabilities

（无现有能力需要修改）

## Impact

- **新增依赖**: OpenCV（视频处理）、httpx/aiohttp（API 调用）、matplotlib/PIL（可视化）
- **API 集成**: Gemini API（原生格式 + OpenAI 兼容格式）
- **输出格式**: 结构化 JSON（环境分析、任务分析、5 个路径点坐标）
