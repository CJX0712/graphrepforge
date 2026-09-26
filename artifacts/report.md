# GraphForge Demo 报告

- 生成时间：2026-09-26T19:49:31+00:00
- 随机种子：42（同 seed 逐字节可复现）
- 总耗时：52.4s
- 成功组合：16；跳过：0

## 基准结果

```
dataset     method      backend     task        metric      value       elapsed     
------------------------------------------------------------------------------------
sbm_demo    spectral    internal    node_classifmacro_f1    0.9686      0.414       
sbm_demo    deepwalk    internal    node_classifmacro_f1    1.0000      10.622      
sbm_demo    node2vec    internal    node_classifmacro_f1    1.0000      10.619      
sbm_demo    grarep      internal    node_classifmacro_f1    0.9686      0.046       
sbm_demo    spectral    internal    link_predictroc_auc     0.9960      0.023       
sbm_demo    deepwalk    internal    link_predictroc_auc     0.9436      10.981      
sbm_demo    node2vec    internal    link_predictroc_auc     0.9436      10.905      
sbm_demo    grarep      internal    link_predictroc_auc     0.9609      0.076       
karate      spectral    internal    node_classifmacro_f1    0.6250      0.007       
karate      deepwalk    internal    node_classifmacro_f1    1.0000      2.315       
karate      node2vec    internal    node_classifmacro_f1    1.0000      2.156       
karate      grarep      internal    node_classifmacro_f1    1.0000      0.021       
karate      spectral    internal    link_predictroc_auc     0.9911      0.010       
karate      deepwalk    internal    link_predictroc_auc     0.8133      2.181       
karate      node2vec    internal    link_predictroc_auc     0.8133      2.054       
karate      grarep      internal    link_predictroc_auc     1.0000      0.041       
```

## Markdown 表

| dataset | method | backend | task | metric | value | elapsed |
|---|---|---|---|---|---|---|
| sbm_demo | spectral | internal | node_classification | macro_f1 | 0.9686 | 0.414 |
| sbm_demo | deepwalk | internal | node_classification | macro_f1 | 1.0000 | 10.622 |
| sbm_demo | node2vec | internal | node_classification | macro_f1 | 1.0000 | 10.619 |
| sbm_demo | grarep | internal | node_classification | macro_f1 | 0.9686 | 0.046 |
| sbm_demo | spectral | internal | link_prediction | roc_auc | 0.9960 | 0.023 |
| sbm_demo | deepwalk | internal | link_prediction | roc_auc | 0.9436 | 10.981 |
| sbm_demo | node2vec | internal | link_prediction | roc_auc | 0.9436 | 10.905 |
| sbm_demo | grarep | internal | link_prediction | roc_auc | 0.9609 | 0.076 |
| karate | spectral | internal | node_classification | macro_f1 | 0.6250 | 0.007 |
| karate | deepwalk | internal | node_classification | macro_f1 | 1.0000 | 2.315 |
| karate | node2vec | internal | node_classification | macro_f1 | 1.0000 | 2.156 |
| karate | grarep | internal | node_classification | macro_f1 | 1.0000 | 0.021 |
| karate | spectral | internal | link_prediction | roc_auc | 0.9911 | 0.010 |
| karate | deepwalk | internal | link_prediction | roc_auc | 0.8133 | 2.181 |
| karate | node2vec | internal | link_prediction | roc_auc | 0.8133 | 2.054 |
| karate | grarep | internal | link_prediction | roc_auc | 1.0000 | 0.041 |
