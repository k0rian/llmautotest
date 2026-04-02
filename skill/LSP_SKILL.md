# LSP Skill

## 目标

该技能用于在代码审计、问题定位、重构评估中提供符号级语义能力。  
它不替代文本搜索，而是补充“定义-引用-诊断-编辑建议”的证据链。

## 适用场景

- 需要确认某个函数/变量的定义来源
- 需要统计符号引用范围与调用影响面
- 需要基于编译器/语言服务诊断定位错误
- 需要评估重命名、格式化、代码动作的改动风险
- 需要跨文件、跨目录检索符号而非纯文本匹配

## 工具分组与职责

### 会话管理

- `lsp_start_session`：启动并初始化语言服务会话
- `lsp_stop_session`：停止会话并释放资源
- `lsp_list_sessions`：查看当前会话列表
- `lsp_get_session_info`：查看会话状态、初始化结果、诊断版本

### 文档同步

- `lsp_open_document`：将文件内容同步到 LSP
- `lsp_change_document`：推送文档变更
- `lsp_save_document`：通知保存，触发相关分析
- `lsp_close_document`：关闭文档同步状态

### 语义查询

- `lsp_hover`：查看当前位置语义信息
- `lsp_definition`：跳转定义
- `lsp_references`：查找引用
- `lsp_document_symbols`：获取当前文件符号树
- `lsp_workspace_symbols`：跨工作区检索符号
- `lsp_completion`：获取补全候选

### 编辑与动作

- `lsp_rename`：评估/执行符号重命名影响
- `lsp_code_actions`：获取可用代码动作
- `lsp_format_document`：获取或执行格式化建议
- `lsp_raw_request`：发送自定义 LSP 请求

### 诊断与观测

- `lsp_get_diagnostics`：读取缓存诊断
- `lsp_wait_for_diagnostics`：等待诊断刷新
- `lsp_get_notifications`：查看近期通知
- `lsp_get_server_logs`：查看服务端日志
- `lsp_wait`：短暂等待异步事件收敛

## 标准执行流程

1. `lsp_start_session` 启动会话并记录 `session_id`
2. `lsp_open_document` 同步目标文件
3. 根据任务调用查询工具：
   - 定位定义：`lsp_definition`
   - 评估影响：`lsp_references`
   - 结构理解：`lsp_document_symbols` / `lsp_workspace_symbols`
4. 需要诊断时调用：
   - `lsp_save_document`
   - `lsp_wait_for_diagnostics`
   - `lsp_get_diagnostics`
5. 必要时读取 `lsp_get_notifications` / `lsp_get_server_logs` 进行排障
6. 结束时 `lsp_close_document` + `lsp_stop_session`

## 证据要求

- 每条结论尽量绑定到文件路径 + 行列位置 + 符号名
- 引用“查询返回结果”而不是口头推测
- 若工具返回为空，明确说明“未检索到”而非默认不存在

## 常见失败与处理

- 会话未启动：先检查 `lsp_list_sessions` 与 `lsp_get_session_info`
- 路径错误：确认文件位于 workspace 且路径合法
- 诊断不更新：执行 `lsp_save_document` 后 `lsp_wait_for_diagnostics`
- 结果不稳定：使用 `lsp_wait` 后重试关键查询

## 输出建议

建议按以下模板输出：

1. 结论
2. 符号级证据（definition / references / diagnostics）
3. 影响分析
4. 修复或重构建议
5. 风险与后续验证点
