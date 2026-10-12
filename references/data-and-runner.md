# 数据治理与计算配置

## 一、将实际数据适配到分析语义

先探索列、样本、键与业务状态，再建立标准分析表。标准字段是接口，不是要求用户的原始文件必须采用这些名称。

| 数据形态 | 适配重点 |
|---|---|
| 商品明细 | 同一订单多行先辨识商品粒度，F按订单去重，金额按合法明细汇总 |
| 订单快照／状态事件 | 先按分析时点重建有效订单，不能将待付、已付、完成三次快照当三笔收入 |
| 支付／退款流水 | 明确支付、撤销、部分退款和调整的含义；订单计数与发生期金额分别归属 |
| 订阅／周期账单 | 明确一次有效续费／账单与订阅主ID的区别，避免所有周期只计一次F |
| 用户行为日志 | 明确有效行为、去重单元／会话和价值代理；非货币M的解释要一致 |
| 客户汇总／快照 | 验证一客一行、同一观察范围、已有R/F/M定义；不假装能从汇总表追溯未提供的明细 |

必要时编写上游适配代码：状态重建、成功行为过滤、币种换算、订单与退款关联、事件去重或多表连接。这属于数据治理阶段，不把不匹配的字段生硬塞进执行器，也不要求用户自己先变成某个项目的数据格式。

ID按标识处理，保留前导零与跨文件一致性。日期解析、时区、缺失、重复、单位、合法极值和金额精度按实际数据检查。只在确认重复定义后删除；变更记录包含人数、行数、行为数、金额及处理理由。

同一人观察时长不同会影响累计F/M，可用统一窗口、资历分层或适合业务的率指标处理；若采用率指标，说明单位并适配计算代码，不将频率与订单次数混称。R可以来自完整历史而F/M只计近期窗口，客户级输入中合法R与F=0可共存；没有可用历史R者另列。交易执行器从同一观察范围构建R/F/M，多个指标窗口先在上游按约定聚合。

## 二、辅助执行器的定位

`scripts/run_segmentation.py`处理已经理解并整理的常见交易表或客户RFM表，计算规则分群、K-means候选／最终画像、基准与交叉结果。完整技能还要完成业务口径判断、最终K选择、解释、命名和运营矩阵。

Python 3.11+及依赖见`scripts/requirements.txt`。优先使用现有环境。脚本读取CSV/XLSX/XLSM，不依赖MySQL；规模／来源超出其支持时，用合适的SQL或分块处理构建标准客户表，再继续分析。额外特征、自定义综合分、复杂金额归因等可生成适配代码实现，不扩展到其他模型。

复制`assets/transactions.example.json`或`assets/customers.example.json`到任务目录并编辑：

```powershell
python "<技能目录>/scripts/run_segmentation.py" --config "<任务目录>/config.json" --output "<任务目录>/run_01"
```

相对输入路径以配置文件所在目录解析。输出目录必须尚不存在，重跑使用新目录。原始数据只读。

## 三、输入与数据口径配置

- `input`：`path`、`level`=`transactions`或`customers`；CSV可配`encoding`，Excel可配`sheet`表名／序号。默认CSV有表头、UTF-8-SIG；其他格式先正确适配。
- `columns`：规范字段名到实际列名的映射。

| 规范字段 | 用途 |
|---|---|
| customer_id | 客户键，两个输入模式均必需 |
| order_id | 有效订单／业务行为单元键，交易模式必需 |
| date | 交易日期，或客户最近有效行为日期 |
| amount | 交易模式的支付／退款发生额；有此字段时优先于数量×单价，不直接接收毛利／成本／非支付额代理值 |
| quantity / unit_price | 两者相乘计算行金额；数量存在时也参与有效购买判断 |
| is_return | 与returns.return_values对应的标识字段 |
| recency_days | 客户汇总的R天数，或由date与reference_date计算 |
| frequency / monetary | 客户汇总模式的F与M |

交易模式至少有customer_id、order_id、date及amount或quantity×unit_price；客户模式至少有customer_id、frequency、monetary及recency_days或date。脚本的客户F为非负整数计数；其他频率定义需适配，不改变真实单位以绕过校验。

`analysis_start/end`可选，含边界当天；省略即当前输入全部范围。客户汇总已由上游按范围计算，不能按最近购买日期过滤来重建交易观察窗口。

`reference_date`为R参考日；交易模式未提供时使用观察范围最后交易日加一天。实际业务快照有指定决策时点时显式配置，不能将技术推定当运营决策日。`currency`仅说明单位，不执行换汇。

清洗与金额：

- `invalid_rows`: `error`／`exclude`，默认error；`drop_exact_duplicates`: 默认false。
- `monetary`: `net`／`gross`；gross在交易执行器中为正向有效销售额，不是毛利。交易amount的符号／零值会参与有效购买判定，不能直接映射毛利、成本或其他非支付额代理。使用这类M时，先按真实成交状态、订单键与日期独立计算R/F，再汇总相应价值为M，通过customers模式输入；这样已成交的负毛利订单仍正确计入F/R。未来若也采用毛利／代理口径，独立聚合对应未来指标并用适配代码连接、比较，不把这些值当支付／退款流水。
- `returns`: 按实际语义配置`prefix`、`return_values`、`signed_amount`或`negative_quantity`，匹配取并集；`amount_sign`=`signed`／`unsigned`控制退款是否需转负。没有某种编码就不要配置它。
- `purchases.allow_zero_amount`: 是否把零价值有效订单计为行为，默认false；已经过上游成功状态过滤的输入才能依此定义有效购买。
- `negative_monetary`: `error`／`exclude`／`keep`；keep保留合法负净值，与特征变换分开。旧signed_log名仅为兼容别名。
- `monetary_zero_tolerance`: 默认1e-9金额单位，只处理微小浮点汇总残差并审计；按最小真实金额精度调小或设0。账务精确计算使用上游整数／Decimal，不用容差掩盖实际差异。
- `order_date_policy`: 同一订单跨多个业务日时选`first`／`last`；默认error提示先确定语义。R/F与月度F使用统一订单日期，M仍按交易发生期。涉及分析分界时先冻结当时订单状态，避免生命周期未来信息改变历史结果。

影响筛选的开关必须用JSON布尔`true`／`false`，不接受字符串`"false"`、0／1或null来代替。包括去重、退货识别、零金额购买、兼容变换开关及未来按月汇总；错误类型应在读取输入前报错。

## 四、评分和标签配置

```json
{
  "rfm": {
    "score_levels": 3,
    "fm_rounding": "none",
    "scoring": {
      "r": {"method": "percentile"},
      "f": {"method": "percentile"},
      "m": {"method": "percentile"}
    },
    "segmentation": {"mode": "generic_matrix"}
  }
}
```

档数L为至少2的整数。每个维度可选`percentile`或`thresholds`；后者需要L−1个有限、严格递增的原单位上界，等于上界归较低原值档，R_days再反向。FM取F/M均值，可选none／floor／round（0.5向上）／ceil。

`generic_matrix`提供中性近期性×历史价值标签，分界可配`cutoffs.low_max/mid_max`。完整工作流必须先参考RFM规则文档中的Shopify完整分组表；采用该映射时配置`score_levels=5`、`fm_rounding=floor`、`mode=shopify`，示例见`examples/shopify.config.json`。`custom`使用规则列表，每条含label及可选r_min/max、f_min/max、m_min/max、fm_min/max；边界包含端点，缺省范围1～L。同一标签可合并多个互斥条件盒子，全部可实现评分组合必须恰好匹配一次。

这些技术配置应由数据探索和业务目的决定。更复杂分群逻辑可在客户级表生成后实现，不将执行器的几种模式当作RFM所有可能的定义。

## 五、K-means配置

- `kmeans.candidates`：候选K列表；未提供时2～9只是初始范围，按样本及不同特征向量数过滤。
- `kmeans.k`：最终K。省略时只完成候选诊断；助手应读画像选择后用新输出目录完成最终版。
- `transforms`：各列R_days／Frequency／Monetary的auto／identity／log1p，Monetary另可signed_log；整个字段也可写auto。
- `scaler`: standard／robust。auto变换以历史非负偏度>1尝试log1p，负M尝试signed_log，其余原值；实际选择及偏度写入audit。助手仍需检查其合理性。
- `random_state`、`n_init`、`silhouette_sample_size`：控制初始化和轮廓抽样，全部可改。
- `seeds`：跨种子ARI必需，默认提供4个非参考种子，去重并剔除参考种子后至少2个有效比较。不同候选K采用同一组比较种子；不将self-comparison的1计作稳定性证据。

候选结果包含每个K的惯性、下降信息、轮廓系数、跨种子ARI汇总、簇大小、原单位画像及候选交叉画像；`candidate_seed_stability.csv`保存ARI明细，`k_selection_diagnostics.png`展示肘部／轮廓／ARI三联图。固定K后另有最终标签、簇画像、交叉结果及`seed_stability.csv`。单一K复现会注明肘部选择证据不足；编号不代表价值顺序。

## 六、基准、交叉与可选未来窗口

`baseline.budgets`接受比例或`{"customers":N}`；比例取floor(客户数×比例)，至少1人。默认几个比例是敏感性起点，`budget_is_business=false`标为情景，不能当实际运营预算。基准为M／F／R排序与固定种子随机选择。

RFM／Cluster行动排序需设置`rfm_priority`／`cluster_priority`，或`historical_priority_rule`=`monetary_desc`／`frequency_desc`／`recency_asc`，也可分别设置`{"rfm":"monetary_desc","cluster":"monetary_desc"}`。历史规则按组历史均值排序，组内客户键稳定破并列；优先级不能从未来结果反推。

缺优先级会标为missing_action_priorities，未选K会标为awaiting_k_selection。**完整分析需补齐排序与最终K，不能把这些中间状态当交付完成。** 若目标无需资源筛选，可用结构比较完成基准阶段并说明选择理由。

`cross.min_customers`是描述性样本提示，默认20，不是显著性保证。交叉输出人数、行／列占比、联合历史画像；有独立结果时增加未来／月度结果。运营矩阵由助手解释生成。

未来窗口可选，只有用户已约定并确认适用时才配置；不能因输入文件中有较晚记录就自动加入：

```json
{
  "holdout": {
    "input": {"path": "later_transactions.csv", "level": "transactions"},
    "columns": {"customer_id": "member", "order_id": "bill", "date": "event_time", "amount": "value"},
    "start": "<结果窗口首日>",
    "end": "<结果窗口末日>",
    "monthly": true
  }
}
```

占位日期需由实际业务填写。省略holdout即当前分层，无须任何未来文件。同源交易文件可省略holdout.input，但须明确非重叠历史截止；客户汇总的未来验证需独立交易输入。未来采用相同事件与价值语义，以冻结历史客户左关联；月度固定客户网格。复杂状态／退款重述先上游处理。

配置了已约定未来窗口时，情景预算也可计算未来等人数表现；没有窗口时基准结果明确为in_sample_descriptive。执行器附带的Oracle列是事后诊断产物，不是可部署策略；是否形成额外Oracle解读按约定交付。跨边界订单生命周期或来源之间重复订单需先治理，audit可能给出qualified，不能将其写成干净时间外证据。

## 七、结果与复现

核心输出包括customer_labels、rfm_summary、cluster_summary、k_diagnostics、candidate_profiles、kmeans_transformed_features、baseline_rankings／metrics／structure／comparisons、交叉人数／占比与联合画像；未来窗口开启时增加客户／分组／交叉未来和月度汇总。

`audit.json`保存输入与脚本哈希、运行版本、有效配置、资格／过滤、变换／缩放、基准与交叉状态、窗口评价类型。阶段状态仅说明计算是否产出；质量解释、最终选择、比较结论与业务落地仍按SKILL完成。

最终按用户需要整理成Notebook、报告、数据表或其他形式。模板数字、分界、预算比例、种子和候选K都是可调整计算参数，不能作为某个行业的默认商业事实。
