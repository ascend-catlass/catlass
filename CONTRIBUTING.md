# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前，请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行[CLA协议签署](https://gitcode.com/cann/community#%E7%AD%BE%E7%BD%B2cla)，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若修改不是简单的bug修复，而是涉及新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过[Issue](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4issue%E5%A4%84%E7%90%86issue%E4%BB%BB%E5%8A%A1)进行方案讨论，以避免代码被拒绝合入。若不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

提交前请在本地完成代码检查，避免因格式或静态检查问题阻塞PR。

首次使用需安装并挂载到 git hook：

```bash
pip install pre-commit
pre-commit install
```

此后每次 `git commit` 都会自动触发检查。也可手动运行：

```bash
pre-commit run --all-files   # 检查全部文件
pre-commit run               # 仅检查已暂存文件
```

开发者贡献场景主要包括：

- Bug修复

  如果在本项目中发现了某些Bug并希望修复，欢迎新建Issue进行反馈和跟踪处理。

  可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，认领该Issue进行处理。

- 贡献新算子样例

  在使用CATLASS的过程中，如果对某算子的参考实现有泛化性增强/性能优化思路，或实现了某类新算子并希望将其贡献为参考样例，欢迎进行样例算子贡献。

  可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对新的样例算子予以说明，并提供设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，认领该Issue进行处理。

  新增算子样例请放置在`examples/`目录下的`${id}_${op_name}`子目录下，其中`${id}`是唯一递增次序，`${op_name}`表示新增算子名称。

  ```bash
  catlass/examples              # 算子样例主目录
  ├── ${id}_${op_name}          # 单算子样例目录
  │   ├── CMakeLists.txt        # CMake编译文件
  │   ├── README.md             # 单算子样例说明文档
  │   └── ${op_name}.cpp        # 主文件
  ...
  ```

  其中`README.md`是对该算子的简要说明，应包含“代码组织”、“使用示例”两方面内容（可参考Matmul的[README文档](https://gitcode.com/cann/catlass/blob/master/examples/00_basic_matmul/README.md)），以便开发者理解。

- 贡献CATLASS代码

  如果对CATLASS代码某一部分有新的特性实现，欢迎在Issue中提出新的想法与设计。

  可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对新的特性设计予以说明，项目成员会进行沟通确认，并在`include/catlass/`下提供合适的位置以贡献特性代码。

  同时，需要在提交的Issue中评论“/assign”或“/assign @yourself”，认领该Issue并在后续完成代码上库。

- 文档纠错

  如果在本项目中发现某些文档描述错误，或内容不够清晰，欢迎新建Issue进行反馈和修复。

  可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，认领该Issue并纠正对应文档描述。

- 帮助解决他人Issue

  如果对社区中他人遇到的问题有合适的解决方法，欢迎在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，可以在Issue评论框中输入“/assign”或“/assign @yourself”，认领该Issue，跟踪协助解决问题。

## 注意事项

- 提交Issue时，需按照 [`.gitcode/ISSUE_TEMPLATE`](.gitcode/ISSUE_TEMPLATE) 中的模板进行提交。若未按照模板提交，Committer 会在评论中提示补充。任何提交后 1 周内仍不满足规范的Issue将会被关闭，并在关闭说明中注明依据。

- 当前，以下组件的贡献将会被低优先级处理，非必要修改将会被直接关闭。若确定需要向以下组件贡献，请务必参加双周 SIG 会议进行讨论。
  - [catlass_cppgen](python/catlass_cppgen/README.md) 原因：catlass_cppgen是CATLASS Python未推出时的Python调用方案，目前已知仅有少量社区项目使用该组件，功能范围有限，且对其修改可能导致这些社区项目无法正常使用。
  - [mstuner_catlass](tools/tuner/README.md) 原因：该调优工具的原理是构建搜索空间，编译大量Kernel然后批量运行获取性能数据，相当于人工一次次地修改样例中的 Tiling，然后使用msprof等工具进行性能采集，以寻找最优 Tiling。目前的AI Agent可以较完美地胜任这种重复性较高的工作，且无需先将算子接入框架；而使用该工具需要先完成算子接入，存在额外工作量。

- 以下情形同样会被关闭，并在关闭说明中注明依据：缺少问题描述、环境信息、复现步骤或日志，且维护者首次回复要求补充后 1 周内仍未补充；与已有 Issue 描述同一问题，关闭并指向原 Issue；问题已在 master 修复，关闭并指向对应提交。

- 上述规则同样适用于 PR；PR 描述中未给出关联 Issue 且未说明原因的，按“信息不足”处理。

- 疑似由脚本批量生成、或同一账号在短时间内提交大量同质 Issue/PR 的，将被整批关闭并注明判定依据，同时上报社区处理。
