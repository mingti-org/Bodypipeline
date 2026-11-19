# 按采集日期划分 collection

新建采集记录带 `collection_date=YYYYMMDD` 和 `batch_id=date-YYYYMMDD`。
日期取所选采集目录末级的日期；非日期自定义目录取创建时的北京时间。
同一连续采集组沿用创建时的日期；跨午夜、重启和延迟上传均不重新计算。
`dataset_name` 保持任务和采集员名称。共享 collection 摘要在已有作用域中加入 batch_id，
保留项目、机器人类型、Profile、模式、格式和存储目标的隔离。
未携带批次字段的旧清单沿用原摘要，禁止自动用上传时间给旧数据补日期。

Client 先读取 `/device/targets` 的 `collection_batch_version`，版本不足则本地保留并重试，
不发起创建上传。创建结果和 COMMITTED 回执均检查 collection 和批次字段。
`/device/uploads/{id}` 状态增加批次、session 和 raw_path 信息，用于迁移后回执核对。

Server 的分组、转换选源及执行校验统一使用 `stored_collection_metadata`。
历史迁移使用有原始摘要校验的 `collection_migration` 记录，不修改已封存 manifest，
因此重传幂等键、原始 manifest_hash、文件 SHA-256 和已有转换快照保持有效。

离线迁移入口：

```bash
PYTHONPATH=. .venv/bin/python scripts/migrate_collection_batches.py --config SERVER_CONFIG --action plan --assignments assignments.json --plan migration-plan.json
PYTHONPATH=. .venv/bin/python scripts/migrate_collection_batches.py --config SERVER_CONFIG --action apply --plan migration-plan.json --journal migration-journal.json
PYTHONPATH=. .venv/bin/python scripts/migrate_collection_batches.py --config SERVER_CONFIG --action rollback --journal migration-journal.json
```

工具持有 Server 的 ProcessLock，服务运行时不能执行。plan/apply 都核对全部源 collection
的已提交采集、Grus session/manifest 摘要、关联任务、文件大小和 SHA-256。
apply 再核对完整计划，写入可恢复日志后同盘改名目录，通过 Store.put 更新上传和关联状态，
更新存在的转换 lineage 路径。转换输出不移动、不重新计算，历史合并结果保留原快照并记录
其来源的新 collection 列表。相关转换尚未结束时拒绝迁移。
异常自动回滚；进程被强制终止时可在同一离线维护窗口执行 rollback。
恢复服务后有新变更时不得直接使用旧日志回滚。

H800 成功迁移并启动后，停止 Grus Client，以原服务的配置及环境执行：

```bash
PYTHONPATH=. .venv/bin/python scripts/migrate_client_collection_batches.py --config CLIENT_CONFIG --assignments assignments.json --journal client-journal.json
```

Client 工具逐条请求 H800 最新状态，核对分组、日期、session、COMMITTED 和摘要后再写入。
同批次的废弃记录同步分组，但不会上传。Client 原始/公开数据路径及封存清单保持不变。
Client 回滚：相同命令加 `--rollback`，无需 assignments。

生产顺序：核对当前两端源码和未提交变更 → 合并并验证 Server（包括该部署特有的列表索引、
缓存、删除/管理 API）→ 停止相关写入并完成 Server 数据备份及迁移 → 启动 Server 并验证能力
与批次列表 → 部署 Client → 停止 Client 同步历史回执 → 恢复 Client 并做只读验收。
不要将另一端或旧 release 的整个目录覆盖当前服务。
