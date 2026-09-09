# Python 包发布

两个包独立发版，向 GitHub 仓库推送对应 tag 后，`publish-pypi.yml` 自动构建并发布到 PyPI。

| Tag | 版本声明 | 发布包 |
| --- | --- | --- |
| `sdk-v<version>` | 根目录 `pyproject.toml` | `alibabacloud-agentcore-sdk` |
| `collaboration-v<version>` | `packages/collaboration/pyproject.toml` | `alibabacloud-agentcore-collaboration` |

## 一次性配置

工作流使用 [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)，不需要把 API Token 存入 GitHub Secrets。

在 GitHub 仓库中创建名为 `pypi` 的 Environment。然后分别进入两个 PyPI 项目的 **Publishing** 页面，添加 GitHub Trusted Publisher：

| 字段 | 值 |
| --- | --- |
| Owner | `ai-agentcore` |
| Repository | `agentcore-sdk-python` |
| Workflow filename | `publish-pypi.yml` |
| Environment | `pypi` |

- [基础包 Publishing 设置](https://pypi.org/manage/project/alibabacloud-agentcore-sdk/settings/publishing/)
- [协作包 Publishing 设置](https://pypi.org/manage/project/alibabacloud-agentcore-collaboration/settings/publishing/)

两个项目都需要配置，单独授权基础包不会授权协作包。GitHub Actions 必须启用；如 Environment 设置了审批人，发布任务会等待批准。建议限制发布 tag 的创建和修改权限。

## 发布步骤

1. 修改要发布的包的 `version`，检查两个包之间的依赖范围，运行相关测试。
2. 将代码及工作流提交并同步到 GitHub。确认 tag 将指向待发布代码，而非旧提交。
3. 创建并推送所需 tag。例如基础包下一补丁版本：

   ```bash
   git tag -a sdk-v0.1.1 -m "Release alibabacloud-agentcore-sdk 0.1.1"
   git push https://github.com/ai-agentcore/agentcore-sdk-python.git refs/tags/sdk-v0.1.1
   ```

   协作包独立发布时使用 `collaboration-v0.1.1`。只推其中一个 tag 不会发布另一个包；若协作包依赖新的基础包版本，应先完成基础包发布。

4. 在 GitHub Actions 中确认构建和发布成功，再检查 PyPI 对应版本及安装结果。

以上命令在 GitHub 对应 checkout 中执行。内部仓库可以保留同名 tag，但只有向 GitHub 推送 tag 会触发此工作流；同步独立历史仓库时，应在各自内容一致的提交上打 tag，不直接推送另一仓库的提交历史。

工作流校验 tag 版本与所选包的声明完全一致，不会自动修改版本。构建任务生成 wheel 和源码包，经过 `twine check --strict` 后，独立发布任务才获取 OIDC 权限并上传。

已发布版本不能覆盖。已有的 `sdk-v0.1.0`、`collaboration-v0.1.0` 不会因为新增工作流而重新触发，也不要移动或重推这些 tag。上传失败后先检查 PyPI 是否已有部分文件，再决定重跑或使用新版本；不要用新内容复用旧版本号。
