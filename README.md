# jioasebanyan

一个基于多 Agent 的小说模拟系统：
- `master agent` 负责世界初始化、POV 观察生成与回合裁定
- `sub agents` 扮演角色进行思考与行动
- `recorder/reviewer` 在模拟后进入成稿与审核循环
- 内置 Chroma 主观记忆库、JSONL 原文档案、Typer CLI

## 环境要求

- Python 3.11+

## 安装

```bash
python -m pip install -e ".[dev]"
```

## 配置

1. 复制环境变量模板并填写你的 API key：

```bash
cp .env.example .env
```

编辑 `.env` 填入 `ANTHROPIC_API_KEY`（或 `OPENAI_API_KEY`）。

2. 复制配置文件并按需调整：

```bash
cp config.example.yaml config.yaml
```

默认 `provider.type=anthropic`，走官方 endpoint。如果你需要：

- **使用代理或自建 endpoint**：在 `config.yaml` 的 `provider.base_url` 填入完整 URL
- **切换到 OpenAI 兼容接口**：把 `provider.type` 改为 `openai`，填好 `base_url` 和 `api_key`

3. （可选）替换模型字符串、调整 `retrieval` 和 `review_loop` 的参数。

## 运行模拟

```bash
novel-sim run --scenario scenarios/inn_rainy_night.yaml --reset
```

## 常用命令

查看当前会话状态：

```bash
novel-sim status
```

查看 trace（例如主 Agent 调用）：

```bash
novel-sim trace master
```

## 测试

```bash
pytest
```

## Web UI（实时看板）

```bash
novel-sim web
```

打开 http://127.0.0.1:8765 ：

- 顶部填场景路径，点 **Init** 初始化（勾 reset 清空旧会话）
- 点 **Run Turn** 推进一回合，能实时看到每个子 Agent 的思考、tool 调用、行动、记忆
- 中间是子 Agent 卡片，右侧是 LLM 调用日志和原始事件流
- 跑够回合后点 **Writing Phase** 进入成稿/审核循环
