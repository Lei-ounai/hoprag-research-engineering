# 证据核验盲审标注规范

每行对应一个案例。用 [盲审材料](annotation_packets.md) 的匿名编号对齐 [标注模板](annotation_template.csv)。建议复制模板后填写。只读原问题、挑战者、比较对象和原排名前十的片段。原报道地址用于出处追溯。评审以所给材料为限，避免使用外部文章或搜索来补齐文本中的缺失事实。

## 按维度独立填写

| 字段 | 可填值 | 裁决标准 |
|---|---|---|
| evidence_support | DIRECT_SUPPORT · PARTIAL_SUPPORT · NO_SUPPORT · CONTRADICTION · UNCERTAIN | 若可见文本完整支持至少一条明确且可单独判断的必要命题，填 DIRECT_SUPPORT，即使整道多跳题还需要其他命题。同一命题只支持背景、论元、时间或限定词的一部分，填 PARTIAL_SUPPORT。主题接近却无法推出任何题面命题填 NO_SUPPORT。原文给出抵触的事实填 CONTRADICTION。段落或题意不足以区分上述状态填 UNCERTAIN。 |
| entity_role | MATCH · MISMATCH · NOT_APPLICABLE · UNCERTAIN | 对齐人物或机构与角色及关系方向。对同一公司，创始人、CEO、投资人和现任负责人可为不同人。仅出现姓名或公司名不够。题目没有相关角色要求时填 NOT_APPLICABLE。 |
| relation_support | SUPPORTED · NEGATED · HEDGED · UNRELATED · UNCERTAIN | 检查主体、谓词、宾语和语气。原文说“被指控”时，题面若只问是否有指控，可以支持该命题。题面问事实是否成立却只见涉嫌、推测或可能时填 HEDGED。另指其他主体、关系或方向填 UNRELATED。明确否定目标关系填 NEGATED。 |
| temporal_scope | CONSISTENT · CONFLICT · UNKNOWN · NOT_APPLICABLE | 对齐题面时间所修饰的动作。使用报道发布日期时仅比较发布时间。事件、任职和成立应找原文中的时间归属。日期缺失、关系词指代不清或相对时间无法锚定填 UNKNOWN。题面无约束填 NOT_APPLICABLE。 |
| temporal_kind | EVENT · PUBLICATION · TENURE · FOUNDING · MULTIPLE · UNKNOWN · NOT_APPLICABLE | 指题面涉及的时间种类。存在多个相关种类填 MULTIPLE，并在理由中分别注明。盲审材料的“日期字面”只列出文本串，作用域由人工确定。 |
| evidence_novelty | NOVEL · REDUNDANT · PARTIAL_NOVELTY · UNCERTAIN | 比较命题层面的事实内容与当前 Top 10 的可见片段。同一事件的赔偿和恢复执照是不同信息。同一内容的改写通常重复。部分新增填 PARTIAL_NOVELTY。基线只有有限片段，无法排除已有时填 UNCERTAIN，并说明范围。 |
| replacement_suitability | REPLACE · KEEP_INCUMBENT · EITHER · UNCERTAIN | 只与该案例展示的比较对象比较。挑战者必须带来清楚的额外价值，并复核替换会失去的事实。有力的新信息仍可能不足以替换持有独有证据的原文档。双方价值相近填 EITHER，无法核对损益填 UNCERTAIN。 |

## 记录理由和见证

- 在 reason 简述核对到的主体、关系、时间种类、出处、确定程度，以及替换的新增信息和损失。需要比较多段时，可在理由里额外写片段编号。
- 在 witness_ref 填片段编号。C1–C3 是挑战者实际评分的可见窗口，I1–I3 是比较对象的可见窗口，B1–B10 是原前十的题面相关节选。
- 在 witness_quote 复制对应可见片段中的连续原文。正向、部分支持或反证必须引自 C 段。对 NO_SUPPORT 或 UNCERTAIN，尽量引与题面最接近的可见句。确实没有可引原文时，witness_ref 填 NONE、引文留空，并在理由注明缺失什么。
- 折叠的完整原始 Chunk 用于定位边界。如果唯一证词在评分窗外，写入理由，并根据 C 段本身标注。截断到半句时也按可见信息判断。
- 直接支持、独立的新颖性与替换价值各自填写。禁止把其中一项的结论默认复制给另一项。

## 示例

假设题面要求判断 Orion 的创始人是否出生于 1975 年，C1 仅写 “Orion CEO Nira joined in 2018”，公司也可能有其他创始人。可标为 NO_SUPPORT、MISMATCH、UNRELATED、UNKNOWN、EVENT，依据原文写清 CEO 与创始人不同、2018 描述入职而非出生。Novelty 与 Replacement 继续检查原前十及给定比较对象。该示例为演示填写而构造。

有两名标注者时先独立完成各自表格，在提交前核对分歧并保留最终审定版。保存为 UTF-8 CSV，维持模板列名。待注或跳过的整行可留空，提交的行须填写所有枚举字段、理由及见证引用。

